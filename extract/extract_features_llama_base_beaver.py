import os
import torch
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
from datasets import load_dataset
from transformers import LlamaForCausalLM, AutoTokenizer

import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "meta-llama/Llama-3.2-3B"  # base (non-instruction) model
ROOT_DIR    = "/data/scratch/acw663/poly-monitor/activation-cache/BeaverTrails/"
os.makedirs(ROOT_DIR, exist_ok=True)

TARGET_LAYERS = [8, 10, 12, 16, 20]
POOL_TYPE     = "mean"
BATCH_SIZE    = 4
MAX_LEN       = 2048

model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
).to(device).eval()
model = torch.compile(model, mode="reduce-overhead")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# hidden dimension (robust to config changes)
D = model.config.hidden_size
NUM_LAYERS = getattr(model.config, "num_hidden_layers", None)
if NUM_LAYERS is not None:
    TARGET_LAYERS = [L for L in TARGET_LAYERS if 0 <= L < NUM_LAYERS]
assert len(TARGET_LAYERS) > 0, "No valid TARGET_LAYERS remain!"

def beavertails_categories(hf_split):
    """Stable, sorted list of category names from the dataset schema."""
    keys = list(hf_split[0]["category"].keys())
    keys.sort()
    return keys

def categories_to_onehot(cat_dicts, category_names):
    """List[Dict[str,bool]] -> (B, C) np.int8 one-hot (multi-label)."""
    idx_map = {k: i for i, k in enumerate(category_names)}
    B, C = len(cat_dicts), len(category_names)
    out = np.zeros((B, C), dtype=np.int8)
    for b, d in enumerate(cat_dicts):
        for k, v in d.items():
            if v:
                out[b, idx_map[k]] = 1
    return out

def preprocess_beaver(batch, category_names):
    enc = tokenizer(
        batch["prompt"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    labels = [0 if safe else 1 for safe in batch["is_safe"]]
    cat_multi = categories_to_onehot(batch["category"], category_names)
    return {
        "input_ids":      enc.input_ids,
        "attention_mask": enc.attention_mask,
        "labels":         labels,
        "cat_multi":      cat_multi,
    }

def prepare_dataset(ds_split, category_names):
    ds = ds_split.map(
        lambda b: preprocess_beaver(b, category_names),
        batched=True, batch_size=256
    )
    return ds.with_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "cat_multi"]
    )

def save_all_layers_one_pass(loader, split_name, num_categories):
    N = len(loader.dataset)

    X_maps = {
        L: open_memmap(
               os.path.join(ROOT_DIR, f"{model_name.split('/')[-1]}-L{L}-{POOL_TYPE}-{split_name}-X.npy"),
               mode="w+", dtype="float32", shape=(N, D)
           )
        for L in TARGET_LAYERS
    }
    base = f"{model_name.split('/')[-1]}-{POOL_TYPE}-{split_name}"
    y_map     = open_memmap(os.path.join(ROOT_DIR, base+"-y.npy"),     mode="w+", dtype="int64", shape=(N,))
    y_cat_map = open_memmap(os.path.join(ROOT_DIR, base+"-y_cat.npy"), mode="w+", dtype="int8",  shape=(N, num_categories))

    activations = {L: None for L in TARGET_LAYERS}
    handles = []
    for L in TARGET_LAYERS:
        def make_hook(idx):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                activations[idx] = h.detach()
            return hook
        handles.append(model.model.layers[L].register_forward_hook(make_hook(L)))

    try:
        idx = 0
        for batch in tqdm(loader, desc=f"{split_name}-all-layers"):
            with torch.inference_mode():
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attn      = batch["attention_mask"].to(device, non_blocking=True)
                _ = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=False)

            bsz = input_ids.size(0)
            seq_lens = (attn.sum(dim=1).clamp(min=1) - 1).cpu()

            for L in TARGET_LAYERS:
                h_gpu = activations[L]
                out   = h_gpu.cpu().to(torch.float32)

                if POOL_TYPE == "finaltoken":
                    vecs = out[torch.arange(bsz), seq_lens]
                elif POOL_TYPE == "mean":
                    mask = attn.unsqueeze(-1).cpu()
                    vecs = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
                elif POOL_TYPE == "max":
                    lo  = out[:, 1:, :]
                    m   = attn[:, 1:].unsqueeze(-1).cpu()
                    vecs= (lo * m).max(1).values
                else:
                    raise ValueError(f"Unknown POOL_TYPE {POOL_TYPE}")

                X_maps[L][idx:idx+bsz] = vecs.numpy()

            end = idx + bsz
            y_map[idx:end]     = batch["labels"].cpu().numpy()
            y_cat_map[idx:end] = batch["cat_multi"].cpu().numpy().astype(np.int8)
            idx = end

    finally:
        for h in handles:
            h.remove()
        for arr in (*X_maps.values(), y_map, y_cat_map):
            arr.flush()

if __name__ == "__main__":
    dataset_name = "PKU-Alignment/BeaverTails"
    hf_train = load_dataset(dataset_name)["330k_train"]
    hf_test  = load_dataset(dataset_name)["330k_test"]

    CATEGORY_NAMES = beavertails_categories(hf_train)
    print(f"{len(CATEGORY_NAMES)} BeaverTails categories:", CATEGORY_NAMES)

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

    save_all_layers_one_pass(train_loader, "330k_train", len(CATEGORY_NAMES))
    save_all_layers_one_pass(test_loader,  "330k_test",  len(CATEGORY_NAMES))