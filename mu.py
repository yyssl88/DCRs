import os
import json
import re
import random
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# =========================================================
# Global Config
# =========================================================
BASE_DIR = "/data_nas/DCR/split_addnoise/pad_mu"

DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "imgs")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

OUT_DIR = Path(os.path.join(DATA_DIR, "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "/data_nas/model_hub/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MU 输出字段
FIELDS = ["border_type", "surface_type", "color_pattern"]

# 噪声配置
NOISE_RATES = [0.00, 0.05, 0.10, 0.15, 0.20]
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

FIELD_OPTIONS = {
    "border_type": ["clear", "unclear"],
    "surface_type": ["smooth", "scaly", "crusted", "ulcerated"],
    "color_pattern": ["uniform", "variegated"]
}

# Prompt
MU_PROMPT = (
    "You are an image analysis assistant.\n\n"
    "Given a close-up image of a skin lesion, do NOT provide any diagnosis.\n"
    "Only describe visible characteristics.\n\n"
    "Extract the following attributes and output strictly in JSON format:\n\n"
    "- border_type: one of [clear, unclear]\n"
    "- surface_type: one of [smooth, scaly, crusted, ulcerated]\n"
    "- color_pattern: one of [uniform, variegated]\n\n"
    "Only output JSON. No explanation."
)

# =========================================================
# Model Loader
# =========================================================
def load_mu_model():
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        use_fast=False
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )
    model.eval()
    return processor, model

# =========================================================
# Utils
# =========================================================
def extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}

def add_noise_to_record(record, noise_rate):
    noisy = record.copy()
    for field, options in FIELD_OPTIONS.items():
        if random.random() < noise_rate:
            candidates = [v for v in options if v != record[field]]
            noisy[field] = random.choice(candidates)
    return noisy

# =========================================================
# MU Inference
# =========================================================
@torch.no_grad()
def run_mu_inference(processor, model):
    df = pd.read_csv(METADATA_PATH)
    assert "img_id" in df.columns

    results = []
    skipped = 0

    for i, row in df.iterrows():
        img_id = str(row["img_id"])
        img_path = os.path.join(IMAGE_DIR, img_id)

        if not os.path.exists(img_path):
            skipped += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": MU_PROMPT},
            ],
        }]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=text,
            images=[image],
            return_tensors="pt"
        ).to(DEVICE)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )

        output_text = processor.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        attrs = extract_json(output_text)

        record = {"img_id": img_id}
        for f in FIELDS:
            record[f] = attrs.get(f, None)

        results.append(record)

        if len(results) % 50 == 0:
            print(f"[MU] processed {len(results)}/{len(df)}")

    print(f"[MU] skipped images: {skipped}")
    return results

# =========================================================
# Save multi-noise outputs
# =========================================================
def save_with_noise(records):
    # 0% noise（原始）
    base_path = OUT_DIR / "mu_pred.jsonl"
    with base_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Saved {base_path}")

    # 多噪声版本
    noise_dir = OUT_DIR / "mu_noise"
    noise_dir.mkdir(exist_ok=True)

    for rate in NOISE_RATES:
        noisy_records = [
            add_noise_to_record(r, rate) for r in records
        ]
        out_path = noise_dir / f"mu_pred_noise_{int(rate*100):02d}.jsonl"
        with out_path.open("w") as f:
            for r in noisy_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ Saved {out_path} (noise={rate*100:.0f}%)")

# =========================================================
# Main
# =========================================================
def main():
    processor, model = load_mu_model()
    records = run_mu_inference(processor, model)
    save_with_noise(records)

if __name__ == "__main__":
    main()
