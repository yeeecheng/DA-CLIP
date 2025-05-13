import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr

embedding_dir = "./embeddings"
blur_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
resize_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # resize levels
jpeg_levels = [10, 20, 30, 40, 50, 60, 70, 80]  # jpeg levels
noisy_levels = [5, 10, 15, 20, 25, 30, 35, 40]  # noisy levels


blur_names = [f"blur{l}_embeddings.npy" for l in blur_levels]
resize_names = [f"resize{l}_embeddings.npy" for l in resize_levels]
jpeg_names = [f"jpeg{l}_embeddings.npy" for l in jpeg_levels]
noisy_names = [f"noisy{l}_embeddings.npy" for l in noisy_levels]

type_name = ["blur", "resize", "jpeg", "noisy"]
type_levels = [blur_levels, resize_levels, jpeg_levels, noisy_levels]

for i, type_names in enumerate([blur_names, resize_names, jpeg_names, noisy_names]):

    # 1. Load all blur embeddings
    type_embeddings = []
    for file in type_names:
        emb = np.load(os.path.join(embedding_dir, file))  # shape: (N, D)
        type_embeddings.append(emb.mean(axis=0))  # take mean vector

    type_embeddings = np.stack(type_embeddings)  # shape: (8, D)

    # 2. Compute pairwise L2 distance matrix
    dist_mat = pairwise_distances(type_embeddings, metric='euclidean')

    # 3. Compute blur level difference matrix
    level_array = np.array(type_levels[i])
    level_diff_mat = np.abs(level_array[:, None] - level_array[None, :])

    # 4. Flatten and compute Spearman correlation
    dist_flat = dist_mat.flatten()
    level_diff_flat = level_diff_mat.flatten()
    corr, _ = spearmanr(dist_flat, level_diff_flat)

    # 5. Visualize
    plt.figure(figsize=(6,5))
    plt.imshow(dist_mat, cmap='viridis')
    plt.colorbar(label='L2 Distance')
    plt.title(f'{type_name[i]} Embedding Distance Matrix\nSpearman Correlation: {corr:.4f}')
    plt.xticks(ticks=range(len(type_levels[i])), labels=type_levels[i])
    plt.yticks(ticks=range(len(type_levels[i])), labels=type_levels[i])
    plt.xlabel(f"{type_name[i]} Level")
    plt.ylabel(f"{type_name[i]} Level")
    plt.savefig(f"./emb_sequence_{type_name[i]}")
    plt.show()
