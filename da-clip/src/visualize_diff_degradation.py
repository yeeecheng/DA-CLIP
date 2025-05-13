import torch
import os
import numpy as np
import open_clip
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

dataset_path = "/mnt/hdd5/yicheng/daclip-uir/universal-image-restoration/datasets/lsdir/test"
classes = [c for c in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, c))]
print(classes)
batch_size = 256  # 每次處理的圖片數量

checkpoint = '/mnt/hdd5/yicheng/daclip-uir/da-clip/src/logs/daclip_ViT-B-32-intra_type_loss_with_max_dist15/checkpoints/epoch_199.pt'
model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=checkpoint)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 用於存儲 embeddings 和 labels
image_embeddings = []
labels = []

# 創建存儲 embeddings 的目錄
embedding_save_path = "./embeddings"
os.makedirs(embedding_save_path, exist_ok=True)

# 遍歷每個類別資料夾，提取 embedding
for class_index, class_name in enumerate(classes):
    print(f"Processing {class_name}...")
    class_path = os.path.join(dataset_path, class_name, "LQ")

    class_embeddings = []
    batch_images = []
    batch_filenames = []

    for filename in tqdm(os.listdir(class_path)):
        image_path = os.path.join(class_path, filename)

        try:
            image = preprocess(Image.open(image_path))
            batch_images.append(image)
            batch_filenames.append(filename)

            # 當圖片數量達到 batch_size，則送入模型推理
            if len(batch_images) >= batch_size:
                batch_tensor = torch.stack(batch_images).to(device)  # 合併 batch
                with torch.no_grad():
                    _, degra_features = model.encode_image(batch_tensor, control=True)
                    degra_features /= degra_features.norm(dim=-1, keepdim=True)

                # 儲存 batch 結果
                embeddings_array = degra_features.cpu().numpy()
                image_embeddings.extend(embeddings_array)
                class_embeddings.extend(embeddings_array)
                labels.extend([class_index] * len(batch_filenames))

                batch_images.clear()
                batch_filenames.clear()

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # 若 batch 中還有剩餘圖片，則處理它們
    if batch_images:
        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            _, degra_features = model.encode_image(batch_tensor, control=True)
            degra_features /= degra_features.norm(dim=-1, keepdim=True)

        embeddings_array = degra_features.cpu().numpy()
        image_embeddings.extend(embeddings_array)
        class_embeddings.extend(embeddings_array)
        labels.extend([class_index] * len(batch_filenames))

    # 存儲該類別的 embeddings
    class_embedding_file = os.path.join(embedding_save_path, f"{class_name}_embeddings.npy")
    np.save(class_embedding_file, np.array(class_embeddings))
    print(f"Saved {class_name} embeddings to {class_embedding_file}")

# 轉為 NumPy 陣列並存儲
image_embeddings = np.array(image_embeddings)
labels = np.array(labels)

# 存儲完整的 embeddings 和 labels
np.save(os.path.join(embedding_save_path, "all_embeddings.npy"), image_embeddings)
np.save(os.path.join(embedding_save_path, "labels.npy"), labels)
print("Saved all embeddings and labels.")

# t-SNE 降維到 2D
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(image_embeddings)

# 繪製 t-SNE 圖
plt.figure(figsize=(8, 6))
for class_index, class_name in enumerate(classes):
    indices = labels == class_index
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=class_name, cmap='viridis', alpha=0.7)

plt.legend()
plt.title("t-SNE Visualization of Image Embeddings")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
output_path = "./tsne_visualization.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"t-SNE visualization saved to {output_path}")
