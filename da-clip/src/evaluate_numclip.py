import torch
import os
import re
from PIL import Image
import open_clip
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from scipy.stats import spearmanr
from scipy.interpolate import make_interp_spline
from sklearn.decomposition import PCA
from open_clip.daclip_model import DaCLIP 
import open_clip


def split_class_name(class_name):
    match = re.match(r'([a-zA-Z]+)([\d.]+)', class_name)
    if match:
        class_type = match.group(1)
        class_level = float(match.group(2))
        return class_type, class_level
    else:
        raise ValueError(f"Unrecognized class name format: {class_name}")

def get_bin_center_bank(model, tokenizer, device):

    degradation_types = ['blur', 'noisy', 'resize', 'jpeg']
    text_feat_bank = {}
    bin_center_bank = {}
    for d_type in degradation_types:
        if d_type in ['blur', 'resize']:
            levels = np.arange(0.5, 4.1, 0.5)
        elif d_type == 'noisy':
            levels = np.arange(5, 41, 5)
        elif d_type == 'jpeg':
            levels = np.arange(10, 81, 10)
        else:
            continue

        # make bins: [(start, end), ...]
        bins = list(zip(levels[:-1], levels[1:]))
        centers = [(s + e) / 2 for s, e in bins]
        # self.bin_center_bank['blur'] → tensor([0.75, 1.25, ..., 3.75])
        bin_center_bank[d_type] = torch.tensor(centers, dtype=torch.float32)

        # semantic prompts
        if d_type == 'blur':
            descriptions = [
                "almost sharp", "slightly blurry", "mildly blurry", "moderately blurry",
                "noticeably blurry", "heavily blurred", "extremely blurry"
            ]
        elif d_type == 'resize':
            descriptions = [
                "nearly original size", "slightly downscaled", "noticeably resized",
                "significantly downscaled", "severely downscaled", "extremely small",
                "barely visible size"
            ]
        elif d_type == 'noisy':
            descriptions = [
                "almost noise-free", "slightly noisy", "mildly noisy", "moderately noisy",
                "noticeably noisy", "heavily noisy", "extremely noisy"
            ]
        elif d_type == 'jpeg':
            descriptions = [
                "high quality jpeg", "slightly compressed jpeg", "noticeably compressed jpeg",
                "moderately compressed jpeg", "heavily compressed jpeg", "very low quality jpeg",
                "extremely compressed jpeg"
            ]
        else:
            descriptions = [f"{d_type} bin {i}" for i in range(len(centers))]


        # self.semantic_prompt_bank['jpeg'] → ["high quality jpeg", ...]
        tokens = tokenizer(descriptions[:len(centers)]).to(device)  # (7, 77)
        batch_tokens = tokens.unsqueeze(0).repeat(32, 1, 1)         # (32, 7, 77)
        flat_tokens = batch_tokens.view(-1, tokens.size(1))         # (224, 77)

        with torch.no_grad():
            text_feats = model.encode_text(flat_tokens, normalize=True)  # (224, D)

        # 再 reshape 回 batch 結構
        text_feats = text_feats.view(32, -1, text_feats.size(-1))        # (32, 7, D)
        text_feat_bank[d_type] = text_feats.mean(dim=0)  # (7, D)

    return bin_center_bank, text_feat_bank


# === 設定參數 ===
dataset_path = "/mnt/hdd5/yicheng/daclip-uir/universal-image-restoration/datasets/lsdir/test"
classes = [c for c in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, c))]
classes.sort()
print(f"Classes: {classes}")
batch_size = 32

base_class_map = {c: re.match(r'[a-zA-Z]+', c).group() for c in classes}
base_classes = sorted(set(base_class_map.values()))
print(f"Base Classes: {base_classes}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def evaluate_checkpoint(checkpoint_path, model_name='daclip_ViT-B-32'):
    # if checkpoint_path.endswith("wild-daclip_ViT-L-14.pt"):
    #     model, preprocess = open_clip.create_model_from_pretrained("daclip_ViT-L-14", pretrained=checkpoint_path)
    #     tokenizer = open_clip.get_tokenizer("ViT-L-14")
    # else:
    #     model, preprocess = open_clip.create_model_from_pretrained(model_name, pretrained=checkpoint_path)
    #     tokenizer = open_clip.get_tokenizer("ViT-B-32")

    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained="laion2b_s34b_b79k")
    preprocess = preprocess_val

    # Step 2: 包裝成 DaCLIP
    model = DaCLIP(clip_model)
    tokenizer = open_clip.get_tokenizer(model_name)

    # Step 3: 載入權重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # 如果是 DDP 存的要去掉 module.
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Step 4: load state_dict
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")




    model.eval().to(device)

    text_full = tokenizer(classes).to(device)
    text_base = tokenizer(base_classes).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features_full = model.encode_text(text_full)
        text_features_full /= text_features_full.norm(dim=-1, keepdim=True)

        text_features_base = model.encode_text(text_base)
        text_features_base /= text_features_base.norm(dim=-1, keepdim=True)

    bin_center_bank, text_feat_bank = get_bin_center_bank(model, tokenizer, device)
    bin_center_bank = {k: v.to(device=device, non_blocking=True) for k, v in bin_center_bank.items()}
    total = 0
    correct = 0
    class_correct = {c: 0 for c in classes}
    class_total = {c: 0 for c in classes}
    base_correct = defaultdict(int)
    base_total = defaultdict(int)

    pred_logs = []  # 儲存 (filename, gt, pred)
    gt_labels = []
    pred_labels = []
    all_preds = []
    all_gts = []
    # all type image degradation embedding
    all_feats = []
    labels = []

    all_feat_gts = []
    all_feat_types = []

    deg_type_to_id = {'blur': 0, 'noisy': 1, 'resize': 2, 'jpeg': 3}

    # 創建存儲 embeddings 的目錄
    embedding_save_path = "./embeddings"
    os.makedirs(embedding_save_path, exist_ok=True)

    for class_index, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name, "LQ")
        batch_images = []
        batch_filenames = []
        class_type, class_level = split_class_name(class_name)
        print(class_level)
        # all class degradation embedding
        class_feats = []
        # deg_type_tensor = torch.tensor([deg_type_to_id[class_type]] * batch_size, device=device)

        for filename in os.listdir(class_path):
            image_path = os.path.join(class_path, filename)
            try:
                image = preprocess(Image.open(image_path).convert("RGB"))
                batch_images.append(image)
                batch_filenames.append(filename)

                if len(batch_images) >= batch_size:
                    batch_tensor = torch.stack(batch_images).to(device)
                    deg_type_tensor = torch.full((len(batch_tensor),), deg_type_to_id[class_type], device=device)
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        _, degra_features = model.encode_image(batch_tensor, control=True)
                        degra_features /= degra_features.norm(dim=-1, keepdim=True)
                        probs_full = (100.0 * degra_features @ text_features_full.T).softmax(dim=-1)
                        preds_full = torch.argmax(probs_full, dim=-1)

                        probs_base = (100.0 * degra_features @ text_features_base.T).softmax(dim=-1)
                        preds_base = torch.argmax(probs_base, dim=-1)

                        pred_val = model.predictor(degra_features, deg_type_tensor, bin_center_bank[class_type], text_feat_bank[class_type])
                        # print(pred_val)
                    all_preds.extend(pred_val.detach().cpu().tolist())
                    all_gts.extend([class_level] * len(pred_val))
                    all_feats.append(degra_features.detach().cpu())
                    class_feats.append(degra_features.detach().cpu())
                    labels.extend([class_index] * len(batch_filenames))

                    all_feat_gts.extend([class_level] * len(degra_features))
                    all_feat_types.extend([class_type] * len(degra_features))

                    for fname, pred_f, pred_b in zip(batch_filenames, preds_full, preds_base):
                        gt = class_name
                        pred = classes[pred_f.item()]
                        pred_logs.append((fname, gt, pred))
                        gt_labels.append(gt)
                        pred_labels.append(pred)

                        total += 1
                        class_total[gt] += 1
                        if pred == gt:
                            correct += 1
                            class_correct[gt] += 1

                        true_base = base_class_map[class_name]
                        pred_base_name = base_classes[pred_b.item()]
                        base_total[true_base] += 1
                        if pred_base_name == true_base:
                            base_correct[true_base] += 1

                    batch_images.clear()
                    batch_filenames.clear()
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        # process remaining images
        if batch_images:
            batch_tensor = torch.stack(batch_images).to(device)
            deg_type_tensor = torch.full((len(batch_tensor),), deg_type_to_id[class_type], device=device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                _, degra_features = model.encode_image(batch_tensor, control=True)
                degra_features /= degra_features.norm(dim=-1, keepdim=True)
                probs_full = (100.0 * degra_features @ text_features_full.T).softmax(dim=-1)
                preds_full = torch.argmax(probs_full, dim=-1)

                probs_base = (100.0 * degra_features @ text_features_base.T).softmax(dim=-1)
                preds_base = torch.argmax(probs_base, dim=-1)

                pred_val = model.predictor(degra_features, deg_type_tensor[:len(batch_tensor)], bin_center_bank[class_type], text_feat_bank[class_type])

            all_preds.extend(pred_val.detach().cpu().tolist())
            all_gts.extend([class_level] * len(pred_val))
            all_feats.append(degra_features.detach().cpu())
            class_feats.append(degra_features.detach().cpu())
            labels.extend([class_index] * len(batch_filenames))
            all_feat_gts.extend([class_level] * len(degra_features))
            all_feat_types.extend([class_type] * len(degra_features))

            for fname, pred_f, pred_b in zip(batch_filenames, preds_full, preds_base):
                gt = class_name
                pred = classes[pred_f.item()]
                pred_logs.append((fname, gt, pred))
                gt_labels.append(gt)
                pred_labels.append(pred)

                total += 1
                class_total[gt] += 1
                if pred == gt:
                    correct += 1
                    class_correct[gt] += 1

                true_base = base_class_map[class_name]
                pred_base_name = base_classes[pred_b.item()]
                base_total[true_base] += 1
                if pred_base_name == true_base:
                    base_correct[true_base] += 1

            batch_images.clear()
            batch_filenames.clear()


        # 存儲該類別的 embeddings
        class_embedding_file = os.path.join(embedding_save_path, f"{class_name}_embeddings.npy")
        class_feats = torch.cat(class_feats, dim=0)  # shape = (N, 512)
        np.save(class_embedding_file, np.array(class_feats))
        print(f"Saved {class_name} embeddings to {class_embedding_file}")


    # === Regression 曲線可視化（分 type） ===
    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    all_feat_types_np = np.array(all_feat_types)

    all_feats = torch.cat(all_feats, dim=0)  # shape = (N, 512)
    np.save(os.path.join(embedding_save_path, "all_embeddings.npy"), np.array(all_feats))
    labels = np.array(labels)
    np.save(os.path.join(embedding_save_path, "labels.npy"), labels)
    print("Saved all embeddings and labels.")


    # for t in sorted(set(all_feat_types)):
    #     idx = (all_feat_types_np == t)
    #     if np.sum(idx) < 10:
    #         continue
    #     gt_sorted = all_gts[idx][np.argsort(all_gts[idx])]
    #     pred_sorted = all_preds[idx][np.argsort(all_gts[idx])]
    #     corr, _ = spearmanr(gt_sorted, pred_sorted)

    #     x_vals = np.linspace(0, len(gt_sorted) - 1, 300)
    #     gt_spline = make_interp_spline(np.arange(len(gt_sorted)), gt_sorted)(x_vals)
    #     pred_spline = make_interp_spline(np.arange(len(pred_sorted)), pred_sorted)(x_vals)

    #     plt.figure(figsize=(10, 4))
    #     plt.plot(x_vals, gt_spline, label='GT', color='gold', linewidth=2)
    #     plt.plot(x_vals, pred_spline, label='Pred', color='orangered', linestyle='--', linewidth=2)
    #     plt.title(f"[{t.upper()}] Regression Curve\n")
    #     # Spearman = {corr:.4f}
    #     plt.xlabel("Sorted Index")
    #     plt.ylabel("Degradation Level")
    #     plt.grid(True, linestyle='--', alpha=0.5)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(f"regression_curve_{t}.png")
    #     plt.close()


    # === Embedding vs Level 的有序性分析 ===
    # all_feats_tensor = torch.cat(all_feats, dim=0).numpy()
    # gt_array = np.array(all_feat_gts)
    # type_array = np.array(all_feat_types)

    # for deg_type in np.unique(type_array):
    #     mask = (type_array == deg_type)
    #     feats = all_feats_tensor[mask]
    #     gts = gt_array[mask]
    #     if len(gts) < 10:
    #         continue
    #     pca = PCA(n_components=1)
    #     proj = pca.fit_transform(feats)[:, 0]
    #     corr, _ = spearmanr(proj, gts)
    #     print(f"📊 Embedding-{deg_type} PCA(1) vs GT correlation: {corr:.4f}")

    #     plt.figure(figsize=(6, 4))
    #     sns.regplot(x=gts, y=proj, scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})
    #     plt.title(f'{deg_type} Embedding PCA(1) vs GT (Spearman r = {corr:.4f})')
    #     plt.xlabel("Ground Truth Degradation Level")
    #     plt.ylabel("PCA(1) Projection")
    #     plt.grid(True, linestyle='--', alpha=0.5)
    #     plt.tight_layout()
    #     plt.savefig(f"embedding_pca_vs_gt_{deg_type}.png")
    #     plt.close()
    #     print(f"📉 PCA vs GT 圖已儲存為 embedding_pca_vs_gt_{deg_type}.png")



    # 儲存 csv
    checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '')
    csv_path = f"predictions_{checkpoint_name}.csv"
    df_pred = pd.DataFrame(pred_logs, columns=["filename", "ground_truth", "prediction"])
    df_pred.to_csv(csv_path, index=False)
    print(f"✅ 預測結果已儲存為 {csv_path}")

    # 混淆矩陣
    cm = confusion_matrix(gt_labels, pred_labels, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix: {checkpoint_name}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{checkpoint_name}.png", dpi=300)
    plt.close()
    print(f"📊 混淆矩陣已儲存為 confusion_matrix_{checkpoint_name}.png")

        # 儲存 regression 預測與真值
    df_reg = pd.DataFrame({
        "pred_val": all_preds,
        "gt_val": all_gts,
        "type": all_feat_types
    })
    reg_csv_path = f"regression_values_{checkpoint_name}.csv"
    df_reg.to_csv(reg_csv_path, index=False)
    print(f"📈 Regression 預測值已儲存為 {reg_csv_path}")


    # 回傳準確率
    full_class_acc = [100.0 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0 for c in classes]
    base_class_acc = [100.0 * base_correct[b] / base_total[b] if base_total[b] > 0 else 0.0 for b in base_classes]
    return base_class_acc, full_class_acc

# === 執行評估 ===
checkpoints = {
    # "wild DACLIP pre-trained": "/mnt/hdd5/yicheng/daclip-uir/weights/wild-daclip_ViT-L-14.pt",
    # "Original CLIP": "/mnt/hdd5/yicheng/daclip-uir/da-clip/src/logs/daclip_ViT-B-32-start_epoch(original_clip)/checkpoints/epoch_1.pt",
    "Our method": "/mnt/hdd5/yicheng/daclip-uir/da-clip/src/logs/daclip_ViT-B-32-countclip_numclip_v2/checkpoints/epoch_183.pt",
    
}

all_base_results = {}
all_full_results = {}

for name, path in checkpoints.items():
    print(f"\nEvaluating {name}")
    base_acc, full_acc = evaluate_checkpoint(path)
    all_base_results[name] = base_acc
    all_full_results[name] = full_acc

plt.figure(figsize=(10, 6))
for name, accs in all_base_results.items():
    plt.plot(base_classes, accs, marker='o', label=name)


df_base = pd.DataFrame(all_base_results, index=base_classes)
df_base.index.name = "Base Class"

plt.figure(figsize=(8, 2 + 0.5 * len(base_classes)))
plt.axis('off')
table = plt.table(cellText=df_base.round(2).values,
                  rowLabels=df_base.index,
                  colLabels=df_base.columns,
                  cellLoc='center',
                  loc='center')
table.scale(1, 1.5)
plt.title("Base Class Accuracy Table")
plt.savefig("base_class_accuracy_table.png", bbox_inches='tight')
plt.show()



plt.figure(figsize=(14, 6))
for name, accs in all_full_results.items():
    plt.plot(classes, accs, marker='o', label=name)

plt.xticks(rotation=90)
plt.title("Full Class Accuracy Comparison Across Checkpoints")
plt.xlabel("Full Class (with strength)")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.legend(title="Checkpoint")
plt.grid(True)
plt.tight_layout()
plt.savefig("compare_full_class_accuracy.png")
plt.show()
