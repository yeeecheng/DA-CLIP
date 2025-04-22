import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# === 設定 ===
root_dir = "/mnt/hdd5/yicheng/daclip-uir/universal-image-restoration/datasets/lsdir/test_center_crop"
degra_prefix = "blur"
levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
# levels = [10, 20, 30, 40, 50, 60]
image_name = "0_image_00000_000000.png"

# === 建立畫布 ===
n = len(levels)
fig, axes = plt.subplots(4, n, figsize=(3*n, 12))

prev_residual = None  # 儲存上一張 residual
for i, level in enumerate(levels):
    folder = f"{degra_prefix}{level}"
    gt_path = os.path.join(root_dir, folder, "GT", image_name)
    lq_path = os.path.join(root_dir, folder, "LQ", image_name)

    # 讀取圖片
    gt = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
    lq = np.array(Image.open(lq_path)).astype(np.float32) / 255.0
    # residual = np.clip(np.abs(lq - gt), 0, 1)
    residual = np.abs(lq - gt)  # 不做 clip，留給 colormap 來視覺化

    # 顯示 LQ
    axes[0, i].imshow(lq)
    axes[0, i].set_title(f"{degra_prefix}{level}")
    axes[0, i].axis('off')

    # 顯示 GT
    axes[1, i].imshow(gt)
    axes[1, i].set_title("GT")
    axes[1, i].axis('off')

    # 顯示 Residual
    axes[2, i].imshow(residual)
    axes[2, i].set_title("Residual")
    axes[2, i].axis('off')

    # 顯示 Residual 差異（Residual_i+1 - Residual_i）
    if prev_residual is not None:
        delta = np.abs(residual - prev_residual)
        delta_gray = delta.mean(axis=-1)  # 灰階化，shape: (H, W)
        vmax = np.percentile(delta_gray, 99)  # robust normalization 避免極端值干擾
        axes[3, i].imshow(delta_gray, cmap='inferno', vmin=0, vmax=vmax)
        axes[3, i].set_title("ΔResidual Heatmap")
    else:
        axes[3, i].axis('off')
        axes[3, i].set_title("ΔResidual")

    axes[3, i].axis('off')
    prev_residual = residual

# 標註 row 標題
row_titles = ["LQ", "GT", "Residual", "ΔResidual"]
for ax, row in zip(axes[:, 0], row_titles):
    ax.set_ylabel(row, fontsize=14)

plt.tight_layout()
plt.savefig("residual_comparison_with_delta.png")
plt.show()
