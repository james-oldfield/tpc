import os
import torch
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True

# ---------------- Config ----------------
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Qwen/Qwen3-30B-A3B-Base"
ROOT_DIR   = "/data/scratch/acw663/poly-monitor/activation-cache/WildGuard/"
os.makedirs(ROOT_DIR, exist_ok=True)

TARGET_LAYERS = [10, 12, 16, 20, 32, 40]
POOL_TYPE     = "mean" 
BATCH_SIZE    = 1
MAX_LEN       = 2048
PIN           = True

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
name2idx = {n:i for i,n in enumerate(sublabels_names)}

# ---------------- Model & Tokenizer ----------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(device).eval()
# model = torch.compile(model, mode="reduce-overhead")

tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

D = model.config.hidden_size
NUM_LAYERS = getattr(model.config, "num_hidden_layers", None)
if NUM_LAYERS is not None:
    TARGET_LAYERS = [L for L in TARGET_LAYERS if 0 <= L < NUM_LAYERS]
assert len(TARGET_LAYERS) > 0, "No valid TARGET_LAYERS remain!"

def preprocess(batch):
    enc = tok(
        batch["prompt"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    # harmful=1, benign/safe=0
    y_bin = [1 if lab == "harmful" else 0 for lab in batch["prompt_harm_label"]]
    y_mc  = [name2idx.get(x, name2idx['others']) for x in batch["subcategory"]]
    return {
        "input_ids":      enc.input_ids,
        "attention_mask": enc.attention_mask,
        "labels":         y_bin,
        "y_multi":        y_mc,
        "adversarial":    batch["adversarial"],
    }

def prepare(ds):
    ds = ds.map(preprocess, batched=True, batch_size=256)
    return ds.with_format(
        type="torch",
        columns=["input_ids","attention_mask","labels","y_multi","adversarial"]
    )

train_ds = prepare(load_dataset("allenai/wildguardmix","wildguardtrain")["train"])
test_ds  = prepare(load_dataset("allenai/wildguardmix","wildguardtest")["test"])

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=PIN, persistent_workers=PIN
)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=PIN, persistent_workers=PIN
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
    # fallback
    for mod in m.modules():
        if hasattr(mod, "layers"):
            try:
                _ = len(mod.layers)
                return mod.layers
            except Exception:
                pass
    raise RuntimeError("Could not locate decoder blocks.")

def save_all_layers_one_pass(loader, split):
    N = len(loader.dataset)

    # memmaps per layer
    X_maps = {
        L: open_memmap(
            os.path.join(ROOT_DIR, f"{model_name.split('/')[-1]}-L{L}-{POOL_TYPE}-{split}-X.npy"),
            mode="w+", dtype="float32", shape=(N, D)
        )
        for L in TARGET_LAYERS
    }
    base = f"{model_name.split('/')[-1]}-{POOL_TYPE}-{split}"
    y_map       = open_memmap(os.path.join(ROOT_DIR, base+"-y.npy"),       "w+", dtype="int64", shape=(N,))
    y_multi_map = open_memmap(os.path.join(ROOT_DIR, base+"-y_multi.npy"), "w+", dtype="int64", shape=(N,))
    y_adv_map   = open_memmap(os.path.join(ROOT_DIR, base+"-y_adv.npy"),   "w+", dtype="int64", shape=(N,))

    blocks = get_decoder_blocks(model)
    acts   = {L: None for L in TARGET_LAYERS}
    handles = []
    for L in TARGET_LAYERS:
        def make_hook(idx):
            def hook(_m, _in, out):
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
            seq_lens = (attn.sum(dim=1).clamp(min=1) - 1).cpu()

            for L in TARGET_LAYERS:
                out = acts[L].cpu().to(torch.float32)  # (B,S,D)
                if POOL_TYPE == "finaltoken":
                    vecs = out[torch.arange(bsz), seq_lens]
                elif POOL_TYPE == "mean":
                    mask = attn.unsqueeze(-1).cpu()
                    vecs = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
                elif POOL_TYPE == "max":
                    lo  = out[:, 1:, :]
                    m   = attn[:, 1:].unsqueeze(-1).cpu()
                    vecs = (lo * m).max(1).values
                else:
                    raise ValueError(f"Unknown POOL_TYPE {POOL_TYPE}")
                X_maps[L][idx:idx+bsz] = vecs.numpy()

            end = idx + bsz
            y_map[idx:end]       = batch["labels"].cpu().numpy()
            y_multi_map[idx:end] = batch["y_multi"].cpu().numpy()
            y_adv_map[idx:end]   = batch["adversarial"].long().cpu().numpy()
            idx = end
    finally:
        for h in handles: h.remove()
        for mm in (*X_maps.values(), y_map, y_multi_map, y_adv_map): mm.flush()

if __name__ == "__main__":
    save_all_layers_one_pass(train_loader, "wildguard_train")
    save_all_layers_one_pass(test_loader,  "wildguard_test")