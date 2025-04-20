import numpy as np
import json
import matplotlib.pyplot as plt
import umap

# 載入資料
embedding_file = "/mnt/HDD7/yicheng/daclip-uir/universal-image-restoration/embeddings/noisy40_embeddings.npy"
json_file = "/mnt/HDD7/yicheng/daclip-uir/universal-image-restoration/datasets/exp/train/noisy40/degraded_prompts.json"

image_embeddings = np.load(embedding_file)

# 載入 blur parameter
with open(json_file, "r") as f:
    degraded_prompts = json.load(f)

blur_parameters = []
image_names = list(degraded_prompts.keys())
for name in image_names:
    prompt = degraded_prompts[name]
    param = int(prompt.split(" ")[-1])
    blur_parameters.append(param)

blur_parameters = np.array(blur_parameters)

# UMAP 降維
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embeddings_umap = reducer.fit_transform(image_embeddings)

# 畫圖
plt.figure(figsize=(8, 6))
scatter = plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], c=blur_parameters, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label="Noisy Parameter (40)")
plt.title("UMAP Visualization of Noisy Embeddings")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

# 儲存圖檔
umap_output = "/mnt/HDD7/yicheng/daclip-uir/universal-image-restoration/umap_noisy40_visualization.png"
plt.savefig(umap_output, dpi=300, bbox_inches="tight")
plt.close()

print(f"UMAP visualization saved to {umap_output}")
