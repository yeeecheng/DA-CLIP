import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# === 讀取儲存的資料 ===
embedding_path = "/mnt/hdd5/yicheng/daclip-uir/da-clip/src/logs/daclip_ViT-B-32-20250421171526/embeddings"
embeddings = np.load(os.path.join(embedding_path, "all_embeddings.npy"))
labels = np.load(os.path.join(embedding_path, "labels.npy"))

# === 類別名稱（與 labels 對應）===
classes = ['blur0.5', 'blur1.0', 'blur1.5', 'blur2.0', 'blur2.5', 'blur3.0', 'blur3.5', 'blur4.0',
           'jpeg10', 'jpeg20', 'jpeg30', 'jpeg40', 'jpeg50', 'jpeg60', 'jpeg70', 'jpeg80',
           'noisy5', 'noisy10', 'noisy15', 'noisy20', 'noisy25', 'noisy30', 'noisy35', 'noisy40',
           'resize0.5', 'resize1.0', 'resize1.5', 'resize2.0', 'resize2.5', 'resize3.0', 'resize3.5', 'resize4.0']

# === Base degradation 顏色（固定同一 base 用同色）===
base_colors = {
    'blur': 'red',
    'jpeg': 'blue',
    'noisy': 'green',
    'resize': 'purple'
}

# === 萃取 base 類別與數值 ===
def extract_base_and_value(name):
    match = re.match(r'([a-zA-Z]+)([0-9.]+)', name)
    if match:
        base, value = match.groups()
        return base, float(value)
    return name, 0.0

# === 分類排序資訊：base → [(label_idx, class_name, value)] ===
from collections import defaultdict
base_groups = defaultdict(list)

for idx, cls in enumerate(classes):
    base, val = extract_base_and_value(cls)
    base_groups[base].append((idx, cls, val))

# 每個 base group 裡面根據 degradation level 排序（決定透明度）
for base in base_groups:
    base_groups[base].sort(key=lambda x: x[2])  # x[2] 是數值大小

# === PCA + t-SNE 降維 ===
pca = PCA(n_components=50)
embeddings_pca = pca.fit_transform(embeddings)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_pca)

# === 繪圖 ===
plt.figure(figsize=(10, 8))

for base, class_list in base_groups.items():
    base_color = base_colors.get(base, 'gray')
    total = len(class_list)

    for rank, (label_idx, class_name, value) in enumerate(class_list):
        indices = labels == label_idx
        x = embeddings_2d[indices, 0]
        y = embeddings_2d[indices, 1]
        alpha = 0.3 + 0.7 * (rank / (total - 1))  # 從 0.3~1.0 線性分布

        plt.scatter(x, y, s=10, color=base_color, alpha=alpha, label=class_name)

# === 圖例只放代表性項目（每個 base 選一個）===
handles = []
used_bases = set()
for base, class_list in base_groups.items():
    if base in used_bases:
        continue
    used_bases.add(base)
    label_idx, class_name, _ = class_list[-1]  # 強度最大的
    dummy = plt.Line2D([], [], marker='o', linestyle='None', color=base_colors[base], label=base)
    handles.append(dummy)

plt.legend(handles=handles, title='Degradation Type', loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("t-SNE: Degradation Type and Level (Color by Type, Transparency by Level)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.tight_layout()
plt.savefig("./tsne_all_classes_fade_by_level.png", dpi=300, bbox_inches='tight')
plt.close()

print("generate tsne_all_classes_fade_by_level.png")
