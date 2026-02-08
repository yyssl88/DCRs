import os
import random
import pandas as pd
import numpy as np

# =========================
# 配置区
# =========================
DATA_DIR = "/data_nas/DCR/split_addnoise/pad_mu/data"
OUT_DIR  = DATA_DIR

RANDOM_SEED = 42
TEST_RATIO = 0.2        # test = 20%
ERROR_RATE = 0.10       # 10% rows in test get noise

IMG_ID_COL = "img_id"
LABEL_COL  = "diagnostic"

# =========================
# 主流程
# =========================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    print(f"📊 原始 metadata: {df.shape}")

    # --------------------------------------------------
    # 1. 按 img_id 拆分（核心修正点 ①）
    # --------------------------------------------------
    img_ids = df[IMG_ID_COL].dropna().unique().tolist()
    random.shuffle(img_ids)

    n_test = int(len(img_ids) * TEST_RATIO)
    test_img_ids  = set(img_ids[:n_test])
    train_img_ids = set(img_ids[n_test:])

    train_clean = df[df[IMG_ID_COL].isin(train_img_ids)].reset_index(drop=True)
    test_clean  = df[df[IMG_ID_COL].isin(test_img_ids)].reset_index(drop=True)

    print(f"✅ train_clean: {train_clean.shape}")
    print(f"✅ test_clean : {test_clean.shape}")

    # --------------------------------------------------
    # 2. 基于 test_clean 构造 test_dirty, 在 diagnostic 列注入 10% 行级噪声
    # --------------------------------------------------
    test_dirty = test_clean.copy(deep=True)

    n_rows = len(test_dirty)
    n_noisy = int(n_rows * ERROR_RATE)

    noisy_indices = np.random.choice(
        test_dirty.index,
        size=n_noisy,
        replace=False
    )

    # diagnostic 的枚举值集合
    diagnostic_values = (
        test_clean[LABEL_COL]
        .dropna()
        .unique()
        .tolist()
    )

    print(f"🧪 注入 diagnostic 噪声: {n_noisy}/{n_rows} ({ERROR_RATE*100:.1f}%)")

    for idx in noisy_indices:
        old_val = test_dirty.at[idx, LABEL_COL]
        candidates = [v for v in diagnostic_values if v != old_val]
        if candidates:
            test_dirty.at[idx, LABEL_COL] = random.choice(candidates)

    # --------------------------------------------------
    # 3. 保存
    # --------------------------------------------------
    train_clean.to_csv(os.path.join(OUT_DIR, "train_clean.csv"), index=False)
    test_clean.to_csv(os.path.join(OUT_DIR, "test_clean.csv"), index=False)
    test_dirty.to_csv(os.path.join(OUT_DIR, "test_dirty.csv"), index=False)

    print("\n🎉 数据集构建完成")
    print(f"📁 输出目录: {OUT_DIR}")
    print(" ├── train_clean.csv")
    print(" ├── test_clean.csv")
    print(" └── test_dirty.csv")


if __name__ == "__main__":
    main()
