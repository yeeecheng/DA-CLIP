import torch
import os
import numpy as np
import open_clip
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# 設定資料夾路徑
image_folder = "./datasets/all/0_000375"
batch_size = 1

# 加載模型
checkpoint = '../weights/wild-daclip_ViT-L-14.pt'
model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-L-14', pretrained=checkpoint)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-L-14')

# 使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 準備儲存 embedding
degradation_embeddings = []
image_embeddings = []
filenames = []

# 收集圖片
batch_images = []
batch_filenames = []

for filename in tqdm(sorted(os.listdir(image_folder))):
    image_path = os.path.join(image_folder, filename)
    try:
        image = preprocess(Image.open(image_path))
        batch_images.append(image)
        batch_filenames.append(filename)

        # 如果達到 batch_size 就送入模型
        if len(batch_images) >= batch_size:
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad():
                image_features, degra_features = model.encode_image(batch_tensor, control=True)
                print(image_features)
                degra_features /= degra_features.norm(dim=-1, keepdim=True)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
            image_embeddings_array = image_features.cpu().numpy()
            image_embeddings.extend(image_embeddings_array)
            degradation_embeddings_array = degra_features.cpu().numpy()
            degradation_embeddings.extend(degradation_embeddings_array)
            filenames.extend(batch_filenames)
            batch_images.clear()
            batch_filenames.clear()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# 處理剩下的圖片
if batch_images:
    batch_tensor = torch.stack(batch_images).to(device)
    with torch.no_grad():
        image_features, degra_features = model.encode_image(batch_tensor, control=True)
        print(image_features)
        degra_features /= degra_features.norm(dim=-1, keepdim=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    image_embeddings_array = image_features.cpu().numpy()
    image_embeddings.extend(image_embeddings_array)
    degradation_embeddings_array = degra_features.cpu().numpy()
    degradation_embeddings.extend(degradation_embeddings_array)
    filenames.extend(batch_filenames)

# 降維
degradation_embeddings = np.array(degradation_embeddings)
image_embeddings = np.array(image_embeddings)

# 計算 Cosine Similarity
cos_sim_matrix = cosine_similarity(image_embeddings)

# 畫 heatmap（可選：限制範圍只看前 100 張）
N = 8
plt.figure(figsize=(10, 8))
sns.heatmap(cos_sim_matrix[:N, :N], cmap="viridis")
plt.title("Cosine Similarity Heatmap (Top {} samples)".format(N))
plt.xlabel("Image Index")
plt.ylabel("Image Index")

# 儲存圖片
heatmap_output = "./cosine_similarity_heatmap.png"
plt.savefig(heatmap_output, dpi=300, bbox_inches="tight")
plt.close()

print(f"Cosine similarity heatmap saved to {heatmap_output}")


##################################################################

tsne = TSNE(n_components=2, perplexity=5, random_state=42)
embeddings_2d = tsne.fit_transform(degradation_embeddings)

# 畫圖
plt.figure(figsize=(8, 6))
for i, (x, y) in enumerate(embeddings_2d):
    plt.scatter(x, y)
    plt.text(x + 0.5, y, filenames[i], fontsize=9)

plt.title("t-SNE of Different Degradations of the Same Image")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
output_path = "./tsne_degradations.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"t-SNE result saved to {output_path}")
