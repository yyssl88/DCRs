import os
import shutil
import pandas as pd

# ==============================
# 路径配置
# ==============================
METADATA_PATH = "metadata.csv"
IMG_SRC_DIR = r"D:\pad\imgs\images"
IMG_DST_DIR = r"D:\pad\imgs\label_imgs"
MU_CSV_PATH = "mu_data.csv"

MAX_PATIENTS = 150

# ==============================
# 创建目标目录
# ==============================
os.makedirs(IMG_DST_DIR, exist_ok=True)

# ==============================
# 读取 metadata
# ==============================
df = pd.read_csv(METADATA_PATH)

assert "patient_id" in df.columns, "metadata.csv 缺少 patient_id 列"
assert "img_id" in df.columns, "metadata.csv 缺少 img_id 列"

# ==============================
# 取前 200 个 patient_id
# ==============================
patient_ids = (
    df["patient_id"]
    .drop_duplicates()
    .head(MAX_PATIENTS)
    .tolist()
)

print(f"选取 patient 数量: {len(patient_ids)}")

# ==============================
# 筛选对应的 img_id
# ==============================
sub_df = df[df["patient_id"].isin(patient_ids)]
img_ids = sub_df["img_id"].tolist()

print(f"匹配到 img 数量: {len(img_ids)}")

# ==============================
# 复制图片 + 收集成功 img_id
# ==============================
copied = 0
missing = []
final_img_ids = []

for img_id in img_ids:
    src_path = os.path.join(IMG_SRC_DIR, img_id)

    # 如果 img_id 不含后缀，尝试常见后缀
    if not os.path.exists(src_path):
        found = False
        for ext in [".jpg", ".png", ".jpeg"]:
            tmp_path = src_path + ext
            if os.path.exists(tmp_path):
                src_path = tmp_path
                img_id = os.path.basename(tmp_path)  # 统一 img_id 为真实文件名
                found = True
                break
        if not found:
            missing.append(img_id)
            continue

    dst_path = os.path.join(IMG_DST_DIR, os.path.basename(src_path))
    shutil.copy2(src_path, dst_path)

    final_img_ids.append(os.path.basename(src_path))
    copied += 1

print(f"成功复制图片: {copied}")
print(f"未找到图片: {len(missing)}")

if missing:
    print("缺失的 img_id 示例:", missing[:10])

# ==============================
# 生成 mu_data.csv（标注表）
# ==============================
mu_df = pd.DataFrame({
    "img_id": final_img_ids,
    "border_type": ["" for _ in final_img_ids],
    "surface_type": ["" for _ in final_img_ids],
    "color_pattern": ["" for _ in final_img_ids],
})

mu_df.to_csv(MU_CSV_PATH, index=False, encoding="utf-8-sig")

print(f"mu_data.csv 已生成: {MU_CSV_PATH}")
print(f"可标注样本数: {len(mu_df)}")
