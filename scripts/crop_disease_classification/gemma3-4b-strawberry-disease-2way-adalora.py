"""
Fine-tune Gemma 3 4B on disease image classification  AdaLoRA + 12 GB VRAM optimised.
- Compatible with RTX 50xx (Blackwell) + PyTorch nightly
- AdaLoRA: dynamically reallocates rank budget across layers during training
- Vision tower frozen  only LM layers fine-tuned
- Lazy dataset: images processed on-the-fly  zero RAM pre-load
"""

import os
import sys
import json
import random
from pathlib import Path
from dotenv import load_dotenv

# Load HuggingFace token from .env file — never hardcode tokens in scripts
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)

os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN", "")
os.environ["HF_TOKEN"]               = os.getenv("HF_TOKEN", "")
HF_TOKEN                             = os.getenv("HF_TOKEN", "")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import bitsandbytes  # noqa: F401  required for adamw_bnb_8bit optimizer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import (
    AdaLoraConfig,
    get_peft_model,
    TaskType,
)

# ─────────────────────────────────────────────
# CONFIG  edit these
# ─────────────────────────────────────────────
DATA_DIR   = "C:/Users/ga_ag494/OneDrive - The University of Akron/crop-scan/data/splits/crop_disease/train/Strawberry"
OUTPUT_DIR = "C:/Users/ga_ag494/OneDrive - The University of Akron/crop-scan/experiments/adalora-gemma3-4b-strawberry-disease-2way"
MODEL_ID   = "google/gemma-3-4b-it"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

NUM_EPOCHS    = 1
BATCH_SIZE    = 1        # must be 1 for 12 GB  do not increase
GRAD_ACCUM    = 1        # no accumulation — maximize step frequency within 6hr limit
LEARNING_RATE = 2e-4     # higher than full fine-tune (2e-5)  adapters train faster
MAX_LENGTH    = 512      # 256 tokens for image + 256 for prompt/response — Gemma3 minimum safe value
MAX_IMAGE_SIZE = 336     # Gemma3 native resolution  do not increase
VAL_SPLIT     = 0.15
SEED          = 42

FREEZE_VISION_TOWER = True   # must be True on 12 GB

# ── AdaLoRA hyperparameters ──────────────────────────────────────────────────
# AdaLoRA starts each layer at ADALORA_INIT_R and prunes toward ADALORA_TARGET_R
# based on singular-value importance scores computed during training.
ADALORA_TARGET_R        = 8    # final avg rank per adapted layer
ADALORA_INIT_R          = 24   # starting rank  must be >= target_r
ADALORA_ALPHA           = 32   # scaling factor (same role as lora_alpha)
ADALORA_DROPOUT         = 0.05
ADALORA_BETA1           = 0.85 # EMA for importance score smoothing
ADALORA_BETA2           = 0.85
ADALORA_ORTH_REG_WEIGHT = 0.5  # orthogonal regularisation strength

ADALORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

SYSTEM_PROMPT = (
    "You are a botanist specializing in plant pathology.\n"
    "Classify each image by identifying whether the strawberry leaf is:\n"
    "1. Healthy or\n"
    "2. LeafScorch\n\n"
    "Focus on key botanical and pathological features such as leaf color, edge burn,\n"
    "spot patterns, texture, and signs of discoloration or desiccation.\n\n"
    "Instructions:\n"
    "For each image, return a classification label, 'Healthy' or 'LeafScorch'.\n\n"
    "Respond clearly in the following format:\n"
    "Classification: [category]"
)


# ─────────────────────────────────────────────
# STEP 1  Discover dataset
# ─────────────────────────────────────────────
def discover_dataset(data_dir):
    data_dir = Path(data_dir)
    assert data_dir.exists(), f"Data directory not found: {data_dir}"

    class_names = sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    assert class_names, "No class subfolders found inside data directory."

    label2id = {name: i for i, name in enumerate(class_names)}
    samples = []
    for cls in class_names:
        for p in (data_dir / cls).rglob("*"):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(p), label2id[cls]))

    print(f"[OK] Found {len(samples)} images across {len(class_names)} classes:")
    for name, idx in label2id.items():
        count = sum(1 for _, l in samples if l == idx)
        print(f"   [{idx}] {name}: {count} images")

    return samples, class_names, label2id


# ─────────────────────────────────────────────
# STEP 2  Disk-cached Dataset
# ─────────────────────────────────────────────
def build_cache(samples, class_names, processor, cache_dir):
    """
    Pre-processes all images once and saves encoded tensors to disk.
    On subsequent runs the cache is reused instantly  no reprocessing.

    Why: processor() + tokenization inside __getitem__ on the main thread
    (dataloader_num_workers=0, required on Windows) takes ~2-3s per sample,
    starving the GPU between steps → 0.01 it/s.
    Pre-caching moves that cost to a one-time setup phase (~5-10 min for
    200 images). After that, __getitem__ is torch.load()  milliseconds.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    already = sum(1 for _ in cache_dir.glob("*.pt"))
    if already == len(samples):
        print(f"[OK] Cache already complete  {len(samples)} tensors in {cache_dir}")
        return

    options     = ", ".join(class_names)
    user_prompt = (
        f"Classify this plant image. "
        f"Choose exactly one from: [{options}]. "
        f"Reply with only the label."
    )

    print(f"[CACHE]  Building tensor cache ({len(samples) - already} remaining) ")
    for idx, (img_path, label_id) in enumerate(samples):
        out_path = cache_dir / f"{idx}.pt"
        if out_path.exists():
            continue

        label_name = class_names[label_id]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        scale = MAX_IMAGE_SIZE / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        messages = [
            {"role": "system",    "content": [{"type": "text",  "text": SYSTEM_PROMPT}]},
            {"role": "user",      "content": [{"type": "image", "image": img},
                                              {"type": "text",  "text": user_prompt}]},
            {"role": "assistant", "content": [{"type": "text",  "text": label_name}]},
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = processor(
            text=text, images=[img], return_tensors="pt",
            padding="max_length", truncation=True, max_length=MAX_LENGTH,
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}

        labels = torch.full_like(enc["input_ids"], -100)
        resp_ids = processor.tokenizer.encode(label_name, add_special_tokens=False)
        inp = enc["input_ids"].tolist()
        rlen = len(resp_ids)
        for i in range(len(inp) - rlen, -1, -1):
            if inp[i : i + rlen] == resp_ids:
                labels[i : i + rlen] = torch.tensor(resp_ids)
                break

        enc["labels"] = labels
        torch.save(enc, out_path)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(samples):
            print(f"   cached {idx + 1}/{len(samples)}")

    print(f"[OK] Cache ready  {len(samples)} tensors in {cache_dir}")


class CachedPlantDataset(Dataset):
    """__getitem__ is a single torch.load()  keeps GPU fully utilised."""

    def __init__(self, samples, cache_dir):
        self.samples   = samples
        self.cache_dir = Path(cache_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.load(self.cache_dir / f"{idx}.pt", weights_only=True)


# ─────────────────────────────────────────────
# STEP 3  Collator
# ─────────────────────────────────────────────
def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ─────────────────────────────────────────────
# STEP 4  Model + AdaLoRA
# ─────────────────────────────────────────────
def load_model_and_processor(model_id, total_steps, hf_token=None):
    """
    total_steps is required by AdaLoraConfig so its rank-budget scheduler
    knows when to finalise rank allocation (happens at ~60% of training).
    """
    token = hf_token or None

    print("\n[LOAD] Loading processor")
    processor = AutoProcessor.from_pretrained(
        model_id, token=token, use_fast=True
    )
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("[LOAD] Loading model in bfloat16")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        token=token,
    )

    # Freeze vision tower  fine-tune language model only
    frozen = trainable_base = 0
    for name, param in model.named_parameters():
        if FREEZE_VISION_TOWER and ("vision_tower" in name or "vision_model" in name):
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
            trainable_base += param.numel()

    # ── AdaLoRA config ────────────────────────────────────────────────────────
    # Unlike LoRA (static rank), AdaLoRA starts at init_r and iteratively
    # reallocates rank budget based on singular-value importance scores,
    # concentrating capacity in the most impactful layers automatically.
    adalora_cfg = AdaLoraConfig(
        target_r=ADALORA_TARGET_R,
        init_r=ADALORA_INIT_R,
        lora_alpha=ADALORA_ALPHA,
        lora_dropout=ADALORA_DROPOUT,
        beta1=ADALORA_BETA1,
        beta2=ADALORA_BETA2,
        orth_reg_weight=ADALORA_ORTH_REG_WEIGHT,
        total_step=total_steps,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=ADALORA_TARGET_MODULES,
    )
    model = get_peft_model(model, adalora_cfg)

    # ── Patch AdaLoRA rank allocator to ignore vision tower layers ────────────
    # Root cause: AdaLoRA attaches importance-score hooks to ALL layers
    # matching target_modules, including vision tower layers that are frozen.
    # Frozen layers receive no gradients, so their importance scores are
    # never populated in exp_avg_ipt. When the rank allocator iterates
    # model.named_parameters() at finalisation, it finds the lora_A keys
    # but crashes with KeyError because the scores don't exist.
    #
    # Fix: monkey-patch _element_score to return a zero tensor for any
    # key not in exp_avg_ipt, rather than raising KeyError.
    # This is safe  a zero importance score means the layer gets pruned
    # to rank 0, which is correct behaviour for a frozen layer.
    from peft.tuners.adalora.layer import RankAllocator
    _original_element_score = RankAllocator._element_score

    def _safe_element_score(self, n):
        if n not in self.exp_avg_ipt or len(self.exp_avg_ipt) == 0:
            return torch.zeros(1, 1, device="cuda")
        return _original_element_score(self, n)

    RankAllocator._element_score = _safe_element_score
    print("[OK] AdaLoRA rank allocator patched  vision tower keys will return zero scores")

    # Print trainable parameter summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"trainable params : {trainable:,}  ({100 * trainable / total:.2f}%)")
    print(f"frozen params    : {total - trainable:,}  ({100 * (total - trainable) / total:.2f}%)")
    print(f"total params     : {total:,}")

    # ── Memory diagnostics ────────────────────────────────────────────────────
    device = next(p for p in model.parameters() if p.requires_grad).device
    assert device.type == "cuda", "[ERROR] Model ended up on CPU  check CUDA setup."

    torch.cuda.empty_cache()
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    vram_used  = torch.cuda.memory_allocated(0) / 1e9
    vram_free  = vram_total - vram_used

    print(f"\n[OK] Model on: {device}")
    print(f"[VRAM] VRAM total : {vram_total:.1f} GB")
    print(f"[VRAM] VRAM used  : {vram_used:.1f} GB  (after model load)")
    print(f"[VRAM] VRAM free  : {vram_free:.1f} GB")

    if vram_total < 11.0:
        raise RuntimeError(
            f"GPU has only {vram_total:.1f} GB total VRAM. "
            "Minimum 12 GB required for Gemma 3 4B AdaLoRA fine-tuning."
        )
    if vram_free < 1.0:
        print(
            f"[WARN]  WARNING: Only {vram_free:.1f} GB VRAM free after model load. "
            "Training may OOM. Try closing other GPU processes first."
        )

    return model, processor


# ─────────────────────────────────────────────
# STEP 5  AdaLoRA-aware Trainer
# ─────────────────────────────────────────────
class AdaLoraTrainer(Trainer):
    """
    Injects the AdaLoRA rank-update step after every optimiser step.

    AdaLoRA maintains a running importance estimate for each singular value
    in every adapted layer. After each step it calls update_and_allocate to:
      1. Update importance scores via EMA.
      2. Prune low-importance singular values (rank budget reallocation).
      3. Zero-out pruned singular values in the weight matrices.

    Skipping this call means no dynamic rank adaptation occurs.
    """

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        # Call on self.model (PeftModel wrapper), not the inner model arg
        if hasattr(self.model, "update_and_allocate"):
            self.model.update_and_allocate(self.state.global_step)

        return loss


# ─────────────────────────────────────────────
# STEP 6  Train
# ─────────────────────────────────────────────
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ── GPU sanity check ──────────────────────────────────────────────────────
    assert torch.cuda.is_available(), (
        "CUDA not available! Install PyTorch with CUDA:\n"
        "pip install --pre torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/nightly/cu128"
    )
    print(f"[GPU]  GPU : {torch.cuda.get_device_name(0)}")
    print(f"[VRAM] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Dataset ───────────────────────────────────────────────────────────────
    samples, class_names, label2id = discover_dataset(DATA_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"class_names": class_names, "label2id": label2id}, f, indent=2)

    labels_only = [l for _, l in samples]
    train_s, val_s = train_test_split(
        samples,
        test_size=VAL_SPLIT,
        stratify=labels_only,
        random_state=SEED,
    )
    print(f"[DATA] Split: {len(train_s)} train / {len(val_s)} val")

    # ── Compute total steps for AdaLoRA rank scheduler ────────────────────────
    steps_per_epoch = max(1, len(train_s) // (BATCH_SIZE * GRAD_ACCUM))
    total_steps     = steps_per_epoch * NUM_EPOCHS
    print(f"[INFO] ~{steps_per_epoch} optimiser steps / epoch  ({total_steps} total)")

    # ── Model ─────────────────────────────────────────────────────────────────
    hf_token = globals().get("HF_TOKEN", None)
    model, processor = load_model_and_processor(MODEL_ID, total_steps, hf_token)

    # ── Pre-cache datasets to disk (runs once, reused on subsequent runs) ─────
    cache_root = Path(OUTPUT_DIR) / "tensor_cache"
    build_cache(train_s, class_names, processor, cache_root / "train")
    build_cache(val_s,   class_names, processor, cache_root / "val")

    train_ds = CachedPlantDataset(train_s, cache_root / "train")
    val_ds   = CachedPlantDataset(val_s,   cache_root / "val")

    torch.cuda.empty_cache()
    print(f"[VRAM] VRAM free before training: {torch.cuda.get_device_properties(0).total_memory / 1e9 - torch.cuda.memory_allocated(0) / 1e9:.1f} GB")

    # ── TrainingArguments — full training (no max_steps cap) ─────────────────
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=-1,                          # -1 = disabled; num_train_epochs controls length
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,  # effective batch = 1
        learning_rate=LEARNING_RATE,
        warmup_steps=10,                       # gentle warmup over first 10 steps
        lr_scheduler_type="cosine",
        bf16=True,
        optim="adamw_bnb_8bit",
        logging_steps=1,                       # log every optimiser step for visibility
        eval_strategy="epoch",                 # evaluate at end of each epoch
        save_strategy="epoch",                 # save checkpoint each epoch
        load_best_model_at_end=True,           # restore best checkpoint after training
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,              # required on Windows
        remove_unused_columns=False,
        seed=SEED,
        gradient_checkpointing=False,
    )

    trainer = AdaLoraTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # stop if val loss stalls
    )

    print("\n[START] AdaLoRA training started  rank budget reallocated dynamically!\n")
    trainer.train()

    # Finalise SVD decomposition before saving
    print("\n[FIX] Finalising AdaLoRA rank allocation")
    trained_model = trainer.model
    trained_model.resize_modules_by_rank_pattern(
        trained_model.peft_config["default"].rank_pattern, "default"
    )

    print("\n[VRAM] Saving AdaLoRA adapter")
    trained_model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"[OK] Done! Model saved to: {OUTPUT_DIR}")


# ─────────────────────────────────────────────
# STEP 7  Inference
# ─────────────────────────────────────────────
def predict(image_path: str, model_dir: str = OUTPUT_DIR):
    from peft import PeftModel

    with open(os.path.join(model_dir, "label_map.json")) as f:
        meta = json.load(f)
    class_names = meta["class_names"]
    options = ", ".join(class_names)

    processor = AutoProcessor.from_pretrained(model_dir)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map={"": 0}, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base, model_dir)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {
                    "type": "text",
                    "text": (
                        f"Classify this plant image. "
                        f"Choose from: [{options}]. "
                        f"Reply with only the label."
                    ),
                },
            ],
        },
    ]

    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[img], return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)

    result = processor.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print(f"Predicted: {result.strip()}")
    return result.strip()


if __name__ == "__main__":
    main()