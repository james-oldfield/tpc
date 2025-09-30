import os
import torch
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor

import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name  = "openai/gpt-oss-20b"
ROOT_DIR    = f"/data/scratch/acw663/poly-monitor/activation-cache/WildGuard/"
os.makedirs(ROOT_DIR, exist_ok=True)

TARGET_LAYERS = [4, 8, 12, 14, 16, 18, 20]
POOL_TYPE     = "mean"
BATCH_SIZE    = 2 
MAX_LEN       = 2048

sublabels_names = [
    'benign', 'causing_material_harm_by_disseminating_misinformation',
    'copyright_violations', 'cyberattack',
    'defamation_encouraging_unethical_or_unsafe_actions',
    'disseminating_false_or_misleading_information_encouraging_disinformation_campaigns',
    'fraud_assisting_illegal_activities', 'mental_health_over-reliance_crisis',
    'others', 'private_information_individual',
    'sensitive_information_organization_government',
    'sexual_content', 'social_stereotypes_and_unfair_discrimination',
    'toxic_language_hate_speech', 'violence_and_physical_harm'
]

processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(device).eval()

model = torch.compile(model)

# dimensions
D = model.config.hidden_size 
NUM_LAYERS = getattr(model.config, "num_hidden_layers", None)

if NUM_LAYERS is not None:
    TARGET_LAYERS = [L for L in TARGET_LAYERS if 0 <= L < NUM_LAYERS]
assert len(TARGET_LAYERS) > 0, "No valid TARGET_LAYERS remain!"

def preprocess(batch):
    messages_list = [[{"role": "user", "content": prompt}] for prompt in batch["prompt"]]

    enc = processor.apply_chat_template(
        messages_list,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

    return {
        "input_ids":      enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels":         [1 if lab == "harmful" else 0 for lab in batch["prompt_harm_label"]],
        "sublabels":      batch["subcategory"],
        "adversarial":    batch["adversarial"],
    }

def prepare_dataset(ds):
    ds = ds.map(preprocess, batched=True, batch_size=128)
    return ds.with_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "sublabels", "adversarial"]
    )

train_ds = prepare_dataset(load_dataset("allenai/wildguardmix", "wildguardtrain")['train'])
test_ds  = prepare_dataset(load_dataset("allenai/wildguardmix", "wildguardtest")['test'])

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True
)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True
)

def get_decoder_blocks(m):
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h
    if hasattr(m, "get_decoder"):
        dec = m.get_decoder()
        if hasattr(dec, "layers"):
            return dec.layers
    candidates = [mod for mod in m.modules() if hasattr(mod, "layers")]
    for c in candidates:
        try:
            _ = len(c.layers)
            return c.layers
        except Exception:
            pass
    raise RuntimeError("Could not locate decoder blocks on this model.")

def save_all_layers_one_pass(loader, split):
    N = len(loader.dataset)

    X_maps = {}
    for L in TARGET_LAYERS:
        base = f"{model_name.split('/')[-1]}-L{L}-{POOL_TYPE}-{split}"
        X_maps[L] = open_memmap(
            os.path.join(ROOT_DIR, base + "-X.npy"),
            mode="w+", dtype="float32", shape=(N, D)
        )
    base = f"{model_name.split('/')[-1]}-{POOL_TYPE}-{split}"
    y_map       = open_memmap(os.path.join(ROOT_DIR, base+"-y.npy"),       mode="w+", dtype="int64", shape=(N,))
    y_multi_map = open_memmap(os.path.join(ROOT_DIR, base+"-y_multi.npy"), mode="w+", dtype="int64", shape=(N,))
    y_adv_map   = open_memmap(os.path.join(ROOT_DIR, base+"-y_adv.npy"),   mode="w+", dtype="int64", shape=(N,))

    blocks  = get_decoder_blocks(model)
    acts    = {L: None for L in TARGET_LAYERS}
    handles = []

    for L in TARGET_LAYERS:
        def make_hook(idx):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                acts[idx] = h.detach()
            return hook
        handles.append(blocks[L].register_forward_hook(make_hook(L)))

    try:
        idx = 0
        for batch in tqdm(loader, desc=f"{split}-all-layers"):
            with torch.inference_mode():
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attn      = batch["attention_mask"].to(device, non_blocking=True)
                _ = model(input_ids=input_ids, attention_mask=attn)

            bsz = input_ids.size(0)
            seq_lens = (attn.sum(dim=1).clamp(min=1) - 1).cpu()  # last real token

            # per-layer pooling + write
            for L in TARGET_LAYERS:
                h_gpu = acts[L]                        # (B, T, D)
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

            # labels once
            end = idx + bsz
            y_map[idx:end]       = batch["labels"].numpy()
            y_multi_map[idx:end] = np.array([sublabels_names.index(s) for s in batch["sublabels"]])
            y_adv_map[idx:end]   = batch["adversarial"].long().numpy()
            idx = end

    finally:
        for h in handles:
            h.remove()
        for mm in (*X_maps.values(), y_map, y_multi_map, y_adv_map):
            mm.flush()

if __name__ == "__main__":
    save_all_layers_one_pass(train_loader, "train")
    save_all_layers_one_pass(test_loader,  "test")