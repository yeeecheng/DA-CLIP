import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# 加載 image embeddings
image_embeddings = np.load("/mnt/hdd7/yicheng/daclip-uir/universal-image-restoration/embeddings/blur37_embeddings.npy")

# 計算 Cosine Similarity
cos_sim_matrix = cosine_similarity(image_embeddings)

# 畫 heatmap（可選：限制範圍只看前 100 張）
N = 10
plt.figure(figsize=(10, 8))
sns.heatmap(cos_sim_matrix[:N, :N], cmap="viridis")
plt.title("Cosine Similarity Heatmap (Top {} samples)".format(N))
plt.xlabel("Image Index")
plt.ylabel("Image Index")

# 儲存圖片
heatmap_output = "./cosine_similarity_heatmap_1.png"
plt.savefig(heatmap_output, dpi=300, bbox_inches="tight")
plt.close()

print(f"Cosine similarity heatmap saved to {heatmap_output}")
