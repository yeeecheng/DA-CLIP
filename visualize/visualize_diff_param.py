import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 設定文件路徑
embedding_file = "/mnt/hdd7/yicheng/daclip-uir/universal-image-restoration/embeddings/jpeg30_embeddings.npy"
json_file = "/mnt/hdd7/yicheng/daclip-uir/universal-image-restoration/datasets/test/train/jpeg30/degraded_prompts.json"

# 加載 embeddings
image_embeddings = np.load(embedding_file)

# 加載 JSON 來獲取每張圖片的模糊參數
with open(json_file, "r") as f:
    degraded_prompts = json.load(f)

# 提取圖像對應的參數
blur_parameters = []
image_names = list(degraded_prompts.keys())  # 取得所有的圖片名稱
for name in image_names:
    prompt = degraded_prompts[name]
    param = float(prompt.split(" ")[-1])  # 提取數字部分
    blur_parameters.append(param)

# 轉換為 NumPy 陣列
blur_parameters = np.array(blur_parameters)

# 使用 t-SNE 降維
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(image_embeddings)

# 根據參數範圍設置顏色
plt.figure(figsize=(8, 6))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=blur_parameters, cmap="viridis", alpha=0.7)
plt.colorbar(scatter, label="jpeg Parameter (30)")
plt.title("t-SNE Visualization of jpeg Embeddings")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

# 存圖
output_path = "./tsne_jpeg30_visualization.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"t-SNE visualization saved to {output_path}")
