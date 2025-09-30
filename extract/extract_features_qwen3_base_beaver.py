import os
import torch
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Qwen/Qwen3-30B-A3B-Base"
ROOT_DIR   = "/data/scratch/acw663/poly-monitor/activation-cache/BeaverTrails/"
os.makedirs(ROOT_DIR, exist_ok=True)

TARGET_LAYERS = [10, 12, 16, 20, 32, 40]
POOL_TYPE     = "mean" 
BATCH_SIZE    = 1
MAX_LEN       = 2048
PIN           = True

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

def beavertails_categories(hf_split):
    keys = list(hf_split[0]["category"].keys())
    keys.sort()
    return keys

def categories_to_onehot(cat_dicts, names):
    idx = {k:i for i,k in enumerate(names)}
    B, C = len(cat_dicts), len(names)
    out = np.zeros((B, C), dtype=np.int8)
    for b, d in enumerate(cat_dicts):
        for k, v in d.items():
            if v:
                out[b, idx[k]] = 1
    return out

def preprocess(batch, cat_names):
    enc = tok(
        batch["prompt"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    # harmful = 1, safe = 0
    y_bin = [0 if safe else 1 for safe in batch["is_safe"]]
    y_cat = categories_to_onehot(batch["category"], cat_names)  # (B,C) np.int8
    return {
        "input_ids":      enc.input_ids,
        "attention_mask": enc.attention_mask,
        "labels":         y_bin,
        "cat_multi":      y_cat,
    }

def prepare(ds_split, cat_names):
    ds = ds_split.map(lambda b: preprocess(b, cat_names), batched=True, batch_size=256)
    return ds.with_format(type="torch", columns=["input_ids","attention_mask","labels","cat_multi"])

def get_decoder_blocks(m):
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h
    if hasattr(m, "get_decoder"):
        dec = m.get_decoder()
        if hasattr(dec, "layers"):
            return dec.layers
    for mod in m.modules():
        if hasattr(mod, "layers"):
            try:
                _ = len(mod.layers)
                return mod.layers
            except Exception:
                pass
    raise RuntimeError("Could not locate decoder blocks.")

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
    y_map     = open_memmap(os.path.join(ROOT_DIR, base+"-y.npy"),     "w+", dtype="int64", shape=(N,))
    y_cat_map = open_memmap(os.path.join(ROOT_DIR, base+"-y_cat.npy"), "w+", dtype="int8",  shape=(N, num_categories))

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
        for batch in tqdm(loader, desc=f"{split_name}-all-layers"):
            with torch.inference_mode():
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attn      = batch["attention_mask"].to(device, non_blocking=True)
                _ = model(input_ids=input_ids, attention_mask=attn)

            bsz = input_ids.size(0)
            seq_lens = (attn.sum(dim=1).clamp(min=1) - 1).cpu()

            for L in TARGET_LAYERS:
                out = acts[L].cpu().to(torch.float32)  # (B,T,D)
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
            y_map[idx:end]     = batch["labels"].cpu().numpy()
            y_cat_map[idx:end] = batch["cat_multi"].cpu().numpy().astype(np.int8)
            idx = end
    finally:
        for h in handles: h.remove()
        for mm in (*X_maps.values(), y_map, y_cat_map): mm.flush()

if __name__ == "__main__":
    ds_name = "PKU-Alignment/BeaverTails"
    hf_train = load_dataset(ds_name)["330k_train"]
    hf_test  = load_dataset(ds_name)["330k_test"]

    CATEGORY_NAMES = beavertails_categories(hf_train)
    print(f"{len(CATEGORY_NAMES)} BeaverTails categories:", CATEGORY_NAMES)

    train_ds = prepare(hf_train, CATEGORY_NAMES)
    test_ds  = prepare(hf_test,  CATEGORY_NAMES)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=PIN, persistent_workers=PIN
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=PIN, persistent_workers=PIN
    )

    save_all_layers_one_pass(train_loader, "330k_train", len(CATEGORY_NAMES))
    save_all_layers_one_pass(test_loader,  "330k_test",  len(CATEGORY_NAMES))