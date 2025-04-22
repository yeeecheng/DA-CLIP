import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr

embedding_dir = "/mnt/hdd5/yicheng/daclip-uir/da-clip/src/logs/daclip_ViT-B-32-contrastive_learning_degraded_emb_2_degraded_emb/embeddings"
blur_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
blur_names = [f"blur{l}_embeddings.npy" for l in blur_levels]

# 1. Load all blur embeddings
blur_embeddings = []
for file in blur_names:
    emb = np.load(os.path.join(embedding_dir, file))  # shape: (N, D)
    blur_embeddings.append(emb.mean(axis=0))  # take mean vector

blur_embeddings = np.stack(blur_embeddings)  # shape: (8, D)

# 2. Compute pairwise L2 distance matrix
dist_mat = pairwise_distances(blur_embeddings, metric='euclidean')

# 3. Compute blur level difference matrix
level_array = np.array(blur_levels)
level_diff_mat = np.abs(level_array[:, None] - level_array[None, :])

# 4. Flatten and compute Spearman correlation
dist_flat = dist_mat.flatten()
level_diff_flat = level_diff_mat.flatten()
corr, _ = spearmanr(dist_flat, level_diff_flat)

# 5. Visualize
plt.figure(figsize=(6,5))
plt.imshow(dist_mat, cmap='viridis')
plt.colorbar(label='L2 Distance')
plt.title(f'Blur Embedding Distance Matrix\nSpearman Correlation: {corr:.4f}')
plt.xticks(ticks=range(len(blur_levels)), labels=blur_levels)
plt.yticks(ticks=range(len(blur_levels)), labels=blur_levels)
plt.xlabel("Blur Level")
plt.ylabel("Blur Level")
plt.savefig("./emb_sequence,png")
plt.show()
