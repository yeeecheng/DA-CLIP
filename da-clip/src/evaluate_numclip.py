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

    bin_center_bank = []
    all_d_type_tokens = []
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
        bin_center_bank.append(torch.tensor(centers, dtype=torch.long, device=device))

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
        for p in descriptions[:len(centers)]:
            tokenized = tokenizer(p)[0].to(device)
            all_d_type_tokens.append(tokenized)

    all_d_type_tokens = torch.stack(all_d_type_tokens)
    bin_center_bank_features = torch.stack(bin_center_bank)
    with torch.no_grad():
        all_d_type_tokens_features = model.encode_text(all_d_type_tokens, normalize=True)  # (224, D)


    return bin_center_bank_features, all_d_type_tokens_features


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
    # print(f"Loaded checkpoint from {checkpoint_path}")
    # print(f"Missing keys: {missing_keys}")
    # print(f"Unexpected keys: {unexpected_keys}")


    model.eval().to(device)

    test_prompt = ["almost sharp", "slightly blurry", "mildly blurry", "moderately blurry",
                "noticeably blurry", "heavily blurred", "extremely blurry", "nearly original size", "slightly downscaled", "noticeably resized",
                "significantly downscaled", "severely downscaled", "extremely small",
                "barely visible size", "almost noise-free", "slightly noisy", "mildly noisy", "moderately noisy",
                "noticeably noisy", "heavily noisy", "extremely noisy","high quality jpeg", "slightly compressed jpeg", "noticeably compressed jpeg",
                "moderately compressed jpeg", "heavily compressed jpeg", "very low quality jpeg",
                "extremely compressed jpeg"]

    text_full = tokenizer(classes).to(device)
    text_base = tokenizer(base_classes).to(device)
    test_prompt = tokenizer(test_prompt).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features_full = model.encode_text(text_full)
        text_features_full /= text_features_full.norm(dim=-1, keepdim=True)

        text_features_base = model.encode_text(text_base)
        text_features_base /= text_features_base.norm(dim=-1, keepdim=True)

        test_prompt_features = model.encode_text(test_prompt)
        test_prompt_features /= test_prompt_features.norm(dim=-1, keepdim=True)

    bin_center_bank_features, all_d_type_tokens_features = get_bin_center_bank(model, tokenizer, device)
    bin_center_bank_features = bin_center_bank_features.to(device=device, non_blocking=True)
    all_d_type_tokens_features = all_d_type_tokens_features.to(device=device, non_blocking=True)
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

                    with torch.no_grad(), torch.cuda.amp.autocast():
                        _, degra_features = model.encode_image(batch_tensor, control=True)
                        degra_features /= degra_features.norm(dim=-1, keepdim=True)
                        probs_full = (100.0 * degra_features @ test_prompt_features.T).softmax(dim=-1)
                        preds_full = torch.argmax(probs_full, dim=-1)

                        probs_base = (100.0 * degra_features @ text_features_base.T).softmax(dim=-1)
                        preds_base = torch.argmax(probs_base, dim=-1)

                        pred_val = model.predictor(degra_features, all_d_type_tokens_features, bin_center_bank_features)
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
                probs_full = (100.0 * degra_features @ test_prompt_features.T).softmax(dim=-1)
                preds_full = torch.argmax(probs_full, dim=-1)

                probs_base = (100.0 * degra_features @ text_features_base.T).softmax(dim=-1)
                preds_base = torch.argmax(probs_base, dim=-1)

                pred_val = model.predictor(degra_features, all_d_type_tokens_features, bin_center_bank_features)

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
        "pred_val_blur": all_preds[:, 0],
        "pred_val_noisy": all_preds[:, 1],
        "pred_val_resize": all_preds[:, 2],
        "pred_val_jpeg": all_preds[:, 3],
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
    "Our method": "/mnt/hdd2/yicheng/daclip-uir/da-clip/src/logs/daclip_ViT-B-32-20250519060311/checkpoints/epoch_98.pt",

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
