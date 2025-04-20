import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm

# 路徑設定（你可以改成你自己的）
embedding_paths = {
    "noisy": "/mnt/hdd7/yicheng/daclip-uir/universal-image-restoration/embeddings/noisy40_embeddings.npy",
    "blur": "/mnt/hdd7/yicheng/daclip-uir/universal-image-restoration/embeddings/blur37_embeddings.npy",
    "resize": "/mnt/hdd7/yicheng/daclip-uir/universal-image-restoration/embeddings/resize40_embeddings.npy",
    "jpeg": "/mnt/hdd7/yicheng/daclip-uir/universal-image-restoration/embeddings/jpeg30_embeddings.npy"
}

# 載入所有 embeddings 並標記對應類別
all_embeddings = []
all_labels = []

for label, path in embedding_paths.items():
    embeddings = np.load(path)
    all_embeddings.append(embeddings)
    all_labels.extend([label] * len(embeddings))

# 合併
all_embeddings = np.vstack(all_embeddings)
all_labels = np.array(all_labels)

# 文字轉數值（為了顏色標記）
label_to_int = {label: idx for idx, label in enumerate(np.unique(all_labels))}
label_ids = np.array([label_to_int[label] for label in all_labels])

# 定義 colormap
cmap = cm.get_cmap('tab10')

# t-SNE 降維
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(all_embeddings)

# 顏色對應每個點
colors = [cmap(label_to_int[label]) for label in all_labels]

# 畫圖
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.7)

# 圖例（legend）
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                      markerfacecolor=cmap(label_to_int[label]), markersize=10)
           for label in label_to_int]
plt.legend(handles=handles, title="Degradation Type", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title("t-SNE Visualization of Different Degradation Types")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

# 存圖
output_path = "./tsne_degradation_types.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"t-SNE visualization saved to {output_path}")
