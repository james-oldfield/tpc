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
ROOT_DIR    = "/data/scratch/acw663/poly-monitor/activation-cache/WildGuard/"
os.makedirs(ROOT_DIR, exist_ok=True)

TARGET_LAYERS = [8, 10, 12, 16, 18, 20, 32, 40]
POOL_TYPE     = "mean"   # one of "finaltoken", "mean", "max"
BATCH_SIZE    = 1
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

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_name,
    trust_remote_code=True,
    output_hidden_states=False,
    torch_dtype=torch.bfloat16,
)
model = model.to(device).eval()

model = torch.compile(model, mode="reduce-overhead")

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# hidden dimension
D = 5376

# 3. Preprocessing
def preprocess_chat(batch):
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

    labels = [1 if lab == "harmful" else 0 for lab in batch["prompt_harm_label"]]

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "sublabels":      batch["subcategory"],
        "adversarial":    batch["adversarial"],
    }

def prepare_dataset(ds):
    ds = ds.map(preprocess_chat, batched=True, batch_size=256)
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

def save_all_layers_one_pass(loader, split):
    N = len(loader.dataset)

    X_maps = {}
    for L in TARGET_LAYERS:
        base = f"{model_name.split('/')[-1]}-L{L}-{POOL_TYPE}-{split}"
        X_maps[L] = open_memmap(
            os.path.join(ROOT_DIR, base + "-X.npy"),
            mode="w+", dtype="float32", shape=(N, D)
        )
    # shared label arrays
    base = f"{model_name.split('/')[-1]}-{POOL_TYPE}-{split}"
    y_map       = open_memmap(os.path.join(ROOT_DIR, base+"-y.npy"),       mode="w+", dtype="int64", shape=(N,))
    y_multi_map = open_memmap(os.path.join(ROOT_DIR, base+"-y_multi.npy"), mode="w+", dtype="int64", shape=(N,))
    y_adv_map   = open_memmap(os.path.join(ROOT_DIR, base+"-y_adv.npy"),   mode="w+", dtype="int64", shape=(N,))

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
        for batch in tqdm(loader, desc=f"{split}-all-layers"):
            with torch.inference_mode():
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attn      = batch["attention_mask"].to(device, non_blocking=True)
                _ = model(input_ids=input_ids, attention_mask=attn)

            bsz = input_ids.size(0)
            seq_lens = (attn.sum(dim=1) - 1).cpu()

            # pool & write per layer
            for L in TARGET_LAYERS:
                h_gpu = activations[L]                  # (B, S, D) on GPU, bfloat16
                out = h_gpu.cpu().to(torch.float32)   # now on CPU float32
                if POOL_TYPE == "finaltoken":
                    # NOTE: gemma3 chat appears to have this specific structure;
                    # w/ final user token at -3 index; note that Gemma has padding as the leading tokens (not at the end); so -3 is okay.
                    # example is like: ['<pad>', '<pad>', ..., '▁final_user_token', '<end_of_turn>', '\n']
                    vecs = out[torch.arange(bsz), -3]
                    #vecs = out[torch.arange(bsz), seq_lens]
                elif POOL_TYPE == "m1finaltoken":
                    vecs = out[torch.arange(bsz), -1]
                elif POOL_TYPE == "m2finaltoken":
                    vecs = out[torch.arange(bsz), -2]
                elif POOL_TYPE == "mean":
                    mask = attn.unsqueeze(-1).cpu()
                    vecs = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
                elif POOL_TYPE == "max":
                    lo = out[:,1:]
                    m  = attn[:,1:].unsqueeze(-1).cpu()
                    vecs = (lo * m).max(1).values
                else:
                    raise ValueError(f"Unknown POOL_TYPE {POOL_TYPE}")

                start, end = idx, idx + bsz
                X_maps[L][start:end] = vecs.numpy()

            # write labels once
            end = idx + bsz
            y_map[idx:end]       = batch["labels"].numpy()
            y_multi_map[idx:end] = np.array(
                [sublabels_names.index(s) for s in batch["sublabels"]]
            )
            y_adv_map[idx:end]   = batch["adversarial"].long().numpy()
            idx = end

    finally:
        for h in handles:
            h.remove()
        for m in (*X_maps.values(), y_map, y_multi_map, y_adv_map):
            m.flush()

if __name__ == "__main__":
    save_all_layers_one_pass(train_loader, "train")
    save_all_layers_one_pass(test_loader,  "test")