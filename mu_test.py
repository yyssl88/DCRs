import os
import json
import random
import re
import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# =========================================================
# Global Config
# =========================================================
BASE_DIR = "/data_nas/DCR/split_addnoise/pad_mu"

DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "label_imgs")
CSV_PATH = os.path.join(DATA_DIR, "mu_data.csv")
GT_PATH = os.path.join(DATA_DIR, "mu_data.csv")

OUT_DIR = os.path.join(DATA_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "/data_nas/model_hub/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FIELDS = ["border_type", "surface_type", "color_pattern"]

MU_FIELDS = {
    "border_type": ["clear", "unclear"],
    "surface_type": ["smooth", "scaly", "crusted", "ulcerated"],
    "color_pattern": ["uniform", "variegated"]
}

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
        use_fast=False   # 稳定优先
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
    """从模型输出中稳健提取 JSON"""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}

# =========================================================
# MU Inference
# =========================================================
def run_mu_inference(processor, model, output_path):
    df = pd.read_csv(CSV_PATH)
    results = []

    for _, row in df.iterrows():
        img_id = row["img_id"]
        img_path = os.path.join(IMAGE_DIR, img_id)
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")

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

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )

        output_text = processor.decode(
            out_ids[0],
            skip_special_tokens=True
        )

        attrs = extract_json(output_text)

        # ===== 工程级兜底：永不丢 img_id =====
        record = {"img_id": img_id}
        for f in FIELDS:
            record[f] = attrs.get(f, None)

        results.append(record)

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[MU] Saved predictions to {output_path}")

# =========================================================
# Evaluation
# =========================================================
def evaluate_mu(pred_path):
    pred = pd.read_json(pred_path, lines=True)
    gt = pd.read_csv(GT_PATH)

    assert "img_id" in pred.columns, f"Pred columns = {pred.columns.tolist()}"

    pred = pred.drop_duplicates(subset=["img_id"], keep="last")
    df = gt.merge(pred, on="img_id", suffixes=("_gt", "_pred"))
    print(f"[Eval] Samples: {len(df)}")

    for field in FIELDS:
        y_true = df[f"{field}_gt"]
        y_pred = df[f"{field}_pred"]

        valid_mask = y_true.notna() & y_pred.notna()
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        if len(y_true) == 0:
            print(f"\n[{field}] No valid samples.")
            continue

        acc = accuracy_score(y_true, y_pred)

        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        print(f"\n[{field}]")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {p:.4f}")
        print(f"Recall   : {r:.4f}")
        print(f"F1-score : {f1:.4f}")

    exact_match = (
        (df["border_type_gt"] == df["border_type_pred"]) &
        (df["surface_type_gt"] == df["surface_type_pred"]) &
        (df["color_pattern_gt"] == df["color_pattern_pred"])
    ).mean()

    print(f"\n[Overall] Exact-Match Accuracy: {exact_match:.4f}")

# =========================================================
# Main
# =========================================================
def main(run_inference=True, evaluate=True):
    base_pred = os.path.join(OUT_DIR, "mu_pred.jsonl")

    if run_inference:
        processor, model = load_mu_model()
        run_mu_inference(processor, model, base_pred)

    if evaluate:
        evaluate_mu(base_pred)

# =========================================================
if __name__ == "__main__":
    main(
        run_inference=True,
        evaluate=True
    )
