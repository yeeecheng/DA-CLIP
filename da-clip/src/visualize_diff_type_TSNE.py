# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from collections import defaultdict

# # 載入儲存的 embeddings 和 labels
# embedding_path = "./embeddings"
# embeddings = np.load(os.path.join(embedding_path, "all_embeddings.npy"))
# labels = np.load(os.path.join(embedding_path, "labels.npy"))

# # 定義類別名稱（順序必須與 label 對應）
# classes = ['blur0.5', 'blur1.0', 'blur1.5', 'blur2.0', 'blur2.5', 'blur3.0', 'blur3.5', 'blur4.0',
#            'jpeg10', 'jpeg20', 'jpeg30', 'jpeg40', 'jpeg50', 'jpeg60', 'jpeg70', 'jpeg80',
#            'noisy5', 'noisy10', 'noisy15', 'noisy20', 'noisy25', 'noisy30', 'noisy35', 'noisy40',
#            'resize0.5', 'resize1.0', 'resize1.5', 'resize2.0', 'resize2.5', 'resize3.0', 'resize3.5', 'resize4.0']

# # 建立 degradation group
# degradation_groups = defaultdict(list)
# for idx, cls in enumerate(classes):
#     base = ''.join([c for c in cls if not c.isdigit() and c != '.'])  # 例如 blur0.5 -> blur
#     degradation_groups[base].append((idx, cls))

# # 先用 PCA 壓到 50 維，再做 t-SNE 降到 2 維（加快速度）
# pca = PCA(n_components=50)
# embeddings_pca = pca.fit_transform(embeddings)
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# embeddings_2d = tsne.fit_transform(embeddings_pca)

# # 為每一個 base type 繪圖
# output_folder = "./tsne_plots_by_type"
# os.makedirs(output_folder, exist_ok=True)

# colors = plt.get_cmap("tab10").colors  # 最多十種顏色，夠用了

# for group_name, group_classes in degradation_groups.items():
#     plt.figure(figsize=(6, 5))

#     for color_idx, (class_idx, class_name) in enumerate(group_classes):
#         indices = labels == class_idx
#         x = embeddings_2d[indices, 0]
#         y = embeddings_2d[indices, 1]
#         plt.scatter(x, y, s=10, alpha=0.6, color=colors[color_idx % len(colors)], label=class_name)

#         # 中心點加文字
#         x_mean, y_mean = np.mean(x), np.mean(y)
#         plt.text(x_mean, y_mean, class_name, fontsize=8, weight='bold')

#     plt.title(f"t-SNE for {group_name} degradations")
#     plt.xlabel("t-SNE Dimension 1")
#     plt.ylabel("t-SNE Dimension 2")
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, f"{group_name}_tsne.png"), dpi=300)
#     plt.close()

# print(f"Saved all t-SNE plots to: {output_folder}")
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict
import re

# --- 載入資料 ---
embedding_path = "./embeddings"
embeddings = np.load(os.path.join(embedding_path, "all_embeddings.npy"))
labels = np.load(os.path.join(embedding_path, "labels.npy"))

# 類別順序
classes = ['blur0.5', 'blur1.0', 'blur1.5', 'blur2.0', 'blur2.5', 'blur3.0', 'blur3.5', 'blur4.0',
           'jpeg10', 'jpeg20', 'jpeg30', 'jpeg40', 'jpeg50', 'jpeg60', 'jpeg70', 'jpeg80',
           'noisy5', 'noisy10', 'noisy15', 'noisy20', 'noisy25', 'noisy30', 'noisy35', 'noisy40',
           'resize0.5', 'resize1.0', 'resize1.5', 'resize2.0', 'resize2.5', 'resize3.0', 'resize3.5', 'resize4.0']

class_to_idx = {cls: i for i, cls in enumerate(classes)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}

# --- 要細分的類別，例如 blur ---
detailed_base = "blur"

# base 類別轉換
def get_base_class(name):
    return re.sub(r'[0-9.]+$', '', name)

# --- 建 label map：細分的（如 blur0.5）保留，其他變成 base label（如 jpeg）
mapped_labels = []
for i in range(len(labels)):
    cls_name = idx_to_class[labels[i]]
    if get_base_class(cls_name) == detailed_base:
        mapped_labels.append(cls_name)  # eg. blur1.5
    else:
        mapped_labels.append(get_base_class(cls_name))  # eg. jpeg, resize, noisy

unique_labels = sorted(set(mapped_labels))
label_to_color = {label: i for i, label in enumerate(unique_labels)}

# --- 降維處理 ---
pca = PCA(n_components=50)
embeddings_pca = pca.fit_transform(embeddings)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings_pca)

# --- 畫圖 ---
plt.figure(figsize=(10, 8))
cmap = plt.get_cmap("tab20")  # 最多20種顏色

for label in unique_labels:
    idx = [i for i, l in enumerate(mapped_labels) if l == label]
    x = embeddings_2d[idx, 0]
    y = embeddings_2d[idx, 1]
    plt.scatter(x, y, s=10, alpha=0.6, label=label, color=cmap(label_to_color[label] % 20))

plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title(f"t-SNE: Detailed '{detailed_base}' Levels + Other Base Categories")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.tight_layout()
plt.savefig(f"./tsne_detailed_{detailed_base}.png", dpi=300)
plt.show()
