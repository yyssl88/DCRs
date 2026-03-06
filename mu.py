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

# MU 输出字段分类：所有字段 vs 需要加噪声的分类字段
ALL_FIELDS = ["reasoning", "border_type", "surface_type", "color_pattern", "artifacts_present"]
FIELDS_TO_NOISE = ["border_type", "surface_type", "color_pattern"]

# 噪声配置
NOISE_RATES = [0.00]
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 更新：加入了 unknown 选项
FIELD_OPTIONS = {
    "border_type": ["clear", "unclear", "unknown"],
    "surface_type": ["smooth", "scaly", "crusted", "ulcerated", "unknown"],
    "color_pattern": ["uniform", "variegated", "unknown"]
}

# 优化后的 Prompt
MU_PROMPT = (
    "You are an expert dermatological image analysis assistant.\n\n"
    "Given a close-up image of a skin lesion, your task is to describe its visual characteristics objectively. "
    "CRITICAL: Do NOT provide any medical diagnosis. Focus ONLY on visible morphological features.\n\n"
    "Analyze the image and extract the following attributes. For each attribute, you MUST select EXACTLY ONE option that represents the MOST DOMINANT feature. Do NOT combine options.\n"
    "- border_type: EXACTLY ONE of [clear, unclear, unknown]\n"
    "- surface_type: EXACTLY ONE of [smooth, scaly, crusted, ulcerated, unknown]\n"
    "- color_pattern: EXACTLY ONE of [uniform, variegated, unknown]\n"
    "- artifacts_present: list any visible image artifacts, such as [hair, ruler, marker, reflection, none]. Output as an array.\n\n"
    "Output STRICTLY in the following JSON format without any markdown code blocks, backticks, or additional explanations:\n"
    "{\n"
    "  \"reasoning\": \"Briefly describe your step-by-step visual observation here before classifying.\",\n"
    "  \"border_type\": \"...\",\n"
    "  \"surface_type\": \"...\",\n"
    "  \"color_pattern\": \"...\",\n"
    "  \"artifacts_present\": []\n"
    "}"
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
    # 1. 优先尝试提取 Markdown 格式的代码块
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.IGNORECASE)
    if match:
        json_str = match.group(1)
    else:
        # 2. 如果没有代码块，找最后出现的一个大括号对
        matches = re.findall(r"\{[^{}]*\}", text, re.S)
        if not matches:
            return {}
        json_str = matches[-1]
    
    try:
        attrs = json.loads(json_str)
        
        # 【新增：防御性清洗】强制处理模型不听话输出多个选项的情况
        # 针对需要加噪声的分类字段，如果模型输出了逗号或 " and "，我们强制只取第一个特征
        for field in ["border_type", "surface_type", "color_pattern"]:
            if field in attrs and isinstance(attrs[field], str):
                val = attrs[field].lower()
                # 切割并只保留第一个词汇
                if "," in val:
                    attrs[field] = val.split(",")[0].strip()
                elif " and " in val:
                    attrs[field] = val.split(" and ")[0].strip()
                    
        return attrs
    except json.JSONDecodeError:
        return {}

def add_noise_to_record(record, noise_rate):
    noisy = record.copy()
    # 仅对分类标签加噪声，reasoning 和 artifacts 保持原样
    for field in FIELDS_TO_NOISE:
        options = FIELD_OPTIONS[field]
        if random.random() < noise_rate:
            # 确保不会把原本就是 None 的值加入到去重逻辑中报错
            current_val = record.get(field)
            candidates = [v for v in options if v != current_val]
            noisy[field] = random.choice(candidates) if candidates else current_val
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

        # 【修改】提升了 max_new_tokens 以容纳 reasoning 输出
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512, 
            do_sample=False
        )

        output_text = processor.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        attrs = extract_json(output_text)

        record = {"img_id": img_id}
        # 【修改】获取并保存所有定义的字段
        for f in ALL_FIELDS:
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
    with base_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Saved {base_path}")

# =========================================================
# Quick Test / Debugging
# =========================================================
def run_quick_test(processor, model, sample_size=20):
    df = pd.read_csv(METADATA_PATH)
    assert "img_id" in df.columns
    
    # 取前 20 条数据进行测试（你也可以用 df.sample(sample_size) 随机抽）
    test_df = df.head(sample_size)
    
    print(f"🚀 [Test] Starting quick test on {len(test_df)} images...\n" + "="*50)
    
    success_count = 0

    for i, row in test_df.iterrows():
        img_id = str(row["img_id"])
        img_path = os.path.join(IMAGE_DIR, img_id)

        if not os.path.exists(img_path):
            print(f"⚠️ [Skip] Image not found: {img_id}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ [Skip] Failed to load image {img_id}: {e}")
            continue

        # 构造输入
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": MU_PROMPT},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt").to(DEVICE)

        # 推理
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        output_text = processor.decode(output_ids[0], skip_special_tokens=True)

        # 解析
        attrs = extract_json(output_text)
        
        # 验证是否成功提取了核心字段
        is_success = all(k in attrs for k in FIELDS_TO_NOISE)
        if is_success:
            success_count += 1

        # 打印对比结果
        print(f"🖼️ Image ID: {img_id}")
        print("-" * 20 + " RAW OUTPUT " + "-" * 20)
        # 截取 Prompt 之后的部分，方便查看模型真实输出
        model_reply = output_text.split("assistant\n")[-1].strip() if "assistant\n" in output_text else output_text
        print(model_reply)
        print("-" * 20 + " EXTRACTED JSON " + "-" * 16)
        print(json.dumps(attrs, indent=2, ensure_ascii=False))
        print("=" * 50 + "\n")

    print(f"🎯 [Test Summary] Extracted JSON successfully for {success_count}/{len(test_df)} images.")

# =========================================================
# Main (Modified for testing)
# =========================================================
def main():
    print("⏳ Loading model...")
    processor, model = load_mu_model()
    
    print("🚀 Starting full MU inference...")
    records = run_mu_inference(processor, model)
    save_with_noise(records)

if __name__ == "__main__":
    main()
