import os
import torch
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name  = "google/gemma-3-27b-it"
SYSTEM_PROMPT = "You are a helpful, honest assistant."
ROOT_DIR    = f"/data/scratch/acw663/poly-monitor/activation-cache/BeaverTrails/"
os.makedirs(ROOT_DIR, exist_ok=True)

TARGET_LAYERS = [8, 10, 12, 16, 18, 20, 32, 40]
POOL_TYPE     = "mean"
BATCH_SIZE    = 2
MAX_LEN       = 2048

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_name,
    trust_remote_code=True,
    output_hidden_states=False,
    torch_dtype=torch.bfloat16,
)
model = model.to(device).eval()
model = torch.compile(model, mode="reduce-overhead")

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# hidden dimension for Gemma-3-27B
D = 5376

def beavertails_categories(hf_split):
    """Derive a stable ordered list of category names from the dataset schema."""
    first = hf_split[0]
    cat_keys = list(first["category"].keys())
    cat_keys.sort()
    return cat_keys

def categories_to_onehot(cat_dicts, category_names):
    """Convert list of dicts (name->bool) to (B, C) int8 one-hot (multi-label)."""
    idx_map = {name: i for i, name in enumerate(category_names)}
    B, C = len(cat_dicts), len(category_names)
    arr = np.zeros((B, C), dtype=np.int8)
    for b, d in enumerate(cat_dicts):
        for k, v in d.items():
            if v:
                j = idx_map[k]
                arr[b, j] = 1
    return arr

def preprocess_chat_beaver(batch, category_names):
    input_ids, attention_mask = [], []
    for prompt in batch["prompt"]:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",   "content": [{"type": "text", "text": prompt}]},
        ]
        enc = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        input_ids.append(enc.input_ids[0].tolist())
        attention_mask.append(enc.attention_mask[0].tolist())

    # binary label: harmful=1, safe=0
    labels = [0 if safe else 1 for safe in batch["is_safe"]]

    # multi-label one-hot for categories
    cat_multi = categories_to_onehot(batch["category"], category_names)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "cat_multi":      cat_multi,
    }

def prepare_dataset(ds_split, category_names):
    return ds_split.map(
        lambda batch: preprocess_chat_beaver(batch, category_names),
        batched=True, batch_size=256
    ).with_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "cat_multi"]
    )

def save_all_layers_one_pass(loader, split_name, num_categories):
    N = len(loader.dataset)

    # prepare memmaps for each layer
    X_maps = {}
    for L in TARGET_LAYERS:
        base = f"{model_name.split('/')[-1]}-L{L}-{POOL_TYPE}-{split_name}"
        X_maps[L] = open_memmap(
            os.path.join(ROOT_DIR, base + "-X.npy"),
            mode="w+", dtype="float32", shape=(N, D)
        )
    # shared label arrays
    base = f"{model_name.split('/')[-1]}-{POOL_TYPE}-{split_name}"
    y_map     = open_memmap(os.path.join(ROOT_DIR, base+"-y.npy"),
                            mode="w+", dtype="int64", shape=(N,))
    y_cat_map = open_memmap(os.path.join(ROOT_DIR, base+"-y_cat.npy"),
                            mode="w+", dtype="int8",  shape=(N, num_categories))

    # set up hooks
    activations = {L: None for L in TARGET_LAYERS}
    handles = []
    for L in TARGET_LAYERS:
        def make_hook(idx):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                activations[idx] = h.detach()  
            return hook
        handles.append(model.get_decoder().layers[L].register_forward_hook(make_hook(L)))

    try:
        idx = 0
        for batch in tqdm(loader, desc=f"{split_name}-all-layers"):
            with torch.inference_mode():
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attn      = batch["attention_mask"].to(device, non_blocking=True)
                _ = model(input_ids=input_ids, attention_mask=attn)

            bsz = input_ids.size(0)
            seq_lens = (attn.sum(dim=1).clamp(min=1) - 1).cpu()

            # pool & write per layer
            for L in TARGET_LAYERS:
                h_gpu = activations[L]                 # (B, S, D) 
                out   = h_gpu.cpu().to(torch.float32)

                if POOL_TYPE == "finaltoken":
                    vecs = out[torch.arange(bsz), seq_lens]
                elif POOL_TYPE == "m1finaltoken":
                    vecs = out[torch.arange(bsz), -1]
                elif POOL_TYPE == "m2finaltoken":
                    vecs = out[torch.arange(bsz), -2]
                elif POOL_TYPE == "mean":
                    mask = attn.unsqueeze(-1).cpu()
                    vecs = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
                elif POOL_TYPE == "max":
                    lo = out[:, 1:, :]
                    m  = attn[:, 1:].unsqueeze(-1).cpu()
                    vecs = (lo * m).max(1).values
                else:
                    raise ValueError(f"Unknown POOL_TYPE {POOL_TYPE}")

                start, end = idx, idx + bsz
                X_maps[L][start:end] = vecs.numpy()

            end = idx + bsz
            y_map[idx:end]     = batch["labels"].cpu().numpy()
            y_cat_map[idx:end] = batch["cat_multi"].cpu().numpy().astype(np.int8)
            idx = end

    finally:
        for h in handles:
            h.remove()
        for m in (*X_maps.values(), y_map, y_cat_map):
            m.flush()

if __name__ == "__main__":
    dataset_name = "PKU-Alignment/BeaverTails"
    hf_train = load_dataset(dataset_name)["330k_train"]
    hf_test  = load_dataset(dataset_name)["330k_test"]

    CATEGORY_NAMES = beavertails_categories(hf_train)
    print(f"{len(CATEGORY_NAMES)} categories:", CATEGORY_NAMES)

    train_ds = prepare_dataset(hf_train, CATEGORY_NAMES)
    test_ds  = prepare_dataset(hf_test,  CATEGORY_NAMES)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # Extract & save
    save_all_layers_one_pass(train_loader, "330k_train", len(CATEGORY_NAMES))
    save_all_layers_one_pass(test_loader,  "330k_test",  len(CATEGORY_NAMES))