import os
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from sklearn.cluster import KMeans

# =========================================================
# Global config
# =========================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BASE_DIR = "/data_nas/DCR/split_addnoise/pad_mu"
DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_CSV = os.path.join(DATA_DIR, "train_clean.csv")
TEST_CLEAN_CSV = os.path.join(DATA_DIR, "test_clean.csv")
TEST_DIRTY_CSV = os.path.join(DATA_DIR, "test_dirty.csv")

IMAGE_DIR = os.path.join(DATA_DIR, "imgs")
OUT_BASE_DIR = os.path.join(DATA_DIR, "mr_outputs_cluster3")
os.makedirs(OUT_BASE_DIR, exist_ok=True)

TARGET_COL = "diagnostic"

N_CLUSTERS = 3
NOISE_RATIOS = [0.00, 0.05, 0.10, 0.15, 0.20]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# Step 1: Load CLIP
# =========================================================
def load_clip():
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    return model, preprocess

# =========================================================
# Step 2: Build CLIP embedding
# =========================================================
@torch.no_grad()
def build_clip_embedding(df, model, preprocess):
    feats = []
    for img_id in df["img_id"]:
        img_path = os.path.join(IMAGE_DIR, img_id)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")
        img = preprocess(Image.open(img_path).convert("RGB")) \
                  .unsqueeze(0).to(DEVICE)
        feat = model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feats.append(feat.cpu().numpy()[0])

    return pd.DataFrame(np.vstack(feats), index=df.index)

# =========================================================
# Step 3: KMeans clustering
# =========================================================
def cluster_embedding(embed_df):
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_SEED,
        n_init="auto"
    )
    labels = kmeans.fit_predict(embed_df.values)
    return pd.Series(
        [f"clip_cluster_{i}" for i in labels],
        index=embed_df.index,
        name="img_visual_cluster"
    )

# =========================================================
# Step 4: Inject cluster-level noise (TRAIN ONLY)
# =========================================================
def inject_cluster_noise(cluster_series, noise_ratio):
    cluster_series = cluster_series.copy()
    n = len(cluster_series)
    n_noise = int(n * noise_ratio)

    if n_noise > 0:
        noise_idx = np.random.choice(
            cluster_series.index,
            size=n_noise,
            replace=False
        )
        all_clusters = cluster_series.unique().tolist()
        for i in noise_idx:
            old = cluster_series.at[i]
            cluster_series.at[i] = np.random.choice(
                [c for c in all_clusters if c != old]
            )

    print(f"[Noise] Cluster noise: {n_noise}/{n} ({noise_ratio*100:.0f}%)")
    return cluster_series

# =========================================================
# Step 5: Run MR pipeline
# =========================================================
def run_mr():
    # --------------------------------------------------
    # ✅ 直接加载“冻结好的数据集”
    # --------------------------------------------------
    train_clean = pd.read_csv(TRAIN_CSV)
    test_clean  = pd.read_csv(TEST_CLEAN_CSV)
    test_dirty  = pd.read_csv(TEST_DIRTY_CSV)

    print("[MR] Loaded datasets:")
    print(f"  train_clean: {train_clean.shape}")
    print(f"  test_clean : {test_clean.shape}")
    print(f"  test_dirty : {test_dirty.shape}")

    model, preprocess = load_clip()

    print("[MR] Building CLIP embeddings")
    embed_train = build_clip_embedding(train_clean, model, preprocess)
    embed_test_clean = build_clip_embedding(test_clean, model, preprocess)
    embed_test_dirty = build_clip_embedding(test_dirty, model, preprocess)

    cluster_train = cluster_embedding(embed_train)
    cluster_test_clean = cluster_embedding(embed_test_clean)
    cluster_test_dirty = cluster_embedding(embed_test_dirty)

    # --------------------------------------------------
    # 遍历不同 cluster noise
    # --------------------------------------------------
    for noise_ratio in NOISE_RATIOS:
        print(f"\n[MR] ===== cluster noise = {noise_ratio:.2f} =====")
        out_dir = os.path.join(
            OUT_BASE_DIR,
            f"noise_{int(noise_ratio*100):02d}"
        )
        os.makedirs(out_dir, exist_ok=True)

        noisy_cluster_train = inject_cluster_noise(
            cluster_train, noise_ratio
        )

        df_train = train_clean.copy()
        df_train["img_visual_cluster"] = noisy_cluster_train.values

        df_test_clean = test_clean.copy()
        df_test_clean["img_visual_cluster"] = cluster_test_clean.values

        df_test_dirty = test_dirty.copy()
        df_test_dirty["img_visual_cluster"] = cluster_test_dirty.values

        df_train.to_csv(
            os.path.join(out_dir, "mr_metadata_train_augmented.csv"),
            index=False
        )
        df_test_clean.to_csv(
            os.path.join(out_dir, "mr_metadata_test_clean_augmented.csv"),
            index=False
        )
        df_test_dirty.to_csv(
            os.path.join(out_dir, "mr_metadata_test_dirty_augmented.csv"),
            index=False
        )

        print(f"[MR] Saved to {out_dir}")

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    run_mr()
