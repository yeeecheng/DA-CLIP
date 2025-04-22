# import torch
# import os
# import re
# from PIL import Image
# import open_clip
# from tqdm import tqdm
# from collections import defaultdict
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns


# dataset_path = "/mnt/hdd5/yicheng/daclip-uir/universal-image-restoration/datasets/lsdir/test"
# classes = [c for c in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, c))]
# classes.sort()
# print(f"Classes: {classes}")
# batch_size = 32

# base_class_map = {c: re.match(r'[a-zA-Z]+', c).group() for c in classes}
# base_classes = sorted(set(base_class_map.values()))
# print(base_class_map)
# print(base_classes)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def evaluate_checkpoint(checkpoint_path, model_name='daclip_ViT-B-32'):

#     if checkpoint_path == "/mnt/hdd5/yicheng/daclip-uir/weights/wild-daclip_ViT-L-14.pt":
#         model, preprocess = open_clip.create_model_from_pretrained("daclip_ViT-L-14", pretrained=checkpoint_path)
#         model.eval()
#         tokenizer = open_clip.get_tokenizer('ViT-L-14')
#         model.to(device)
#     else:
#         model, preprocess = open_clip.create_model_from_pretrained(model_name, pretrained=checkpoint_path)
#         model.eval()
#         tokenizer = open_clip.get_tokenizer('ViT-B-32')
#         model.to(device)

#     text_full = tokenizer(classes).to(device)
#     text_base = tokenizer(base_classes).to(device)

#     with torch.no_grad(), torch.cuda.amp.autocast():
#         text_features_full = model.encode_text(text_full)
#         text_features_full /= text_features_full.norm(dim=-1, keepdim=True)

#         text_features_base = model.encode_text(text_base)
#         text_features_base /= text_features_base.norm(dim=-1, keepdim=True)

#     total = 0
#     correct = 0
#     class_correct = {c: 0 for c in classes}
#     class_total = {c: 0 for c in classes}
#     base_correct = defaultdict(int)
#     base_total = defaultdict(int)

#     for class_index, class_name in enumerate(classes):
#         class_path = os.path.join(dataset_path, class_name, "LQ")
#         batch_images = []

#         filenames = os.listdir(class_path)
#         for filename in filenames:
#             image_path = os.path.join(class_path, filename)
#             try:
#                 image = preprocess(Image.open(image_path).convert("RGB"))
#                 batch_images.append(image)

#                 if len(batch_images) >= batch_size:
#                     batch_tensor = torch.stack(batch_images).to(device)
#                     with torch.no_grad(), torch.cuda.amp.autocast():
#                         _, degra_features = model.encode_image(batch_tensor, control=True)
#                         degra_features /= degra_features.norm(dim=-1, keepdim=True)

#                         probs_full = (100.0 * degra_features @ text_features_full.T).softmax(dim=-1)
#                         preds_full = torch.argmax(probs_full, dim=-1)

#                         probs_base = (100.0 * degra_features @ text_features_base.T).softmax(dim=-1)
#                         preds_base = torch.argmax(probs_base, dim=-1)

#                     for pred_f, pred_b in zip(preds_full, preds_base):
#                         total += 1
#                         class_total[class_name] += 1
#                         if pred_f.item() == class_index:
#                             correct += 1
#                             class_correct[class_name] += 1

#                         true_base = base_class_map[class_name]
#                         pred_base_name = base_classes[pred_b.item()]
#                         base_total[true_base] += 1
#                         if pred_base_name == true_base:
#                             base_correct[true_base] += 1

#                     batch_images.clear()
#             except Exception as e:
#                 print(f"Error processing {image_path}: {e}")

#         if batch_images:
#             batch_tensor = torch.stack(batch_images).to(device)
#             with torch.no_grad(), torch.cuda.amp.autocast():
#                 _, degra_features = model.encode_image(batch_tensor, control=True)
#                 degra_features /= degra_features.norm(dim=-1, keepdim=True)

#                 probs_full = (100.0 * degra_features @ text_features_full.T).softmax(dim=-1)
#                 preds_full = torch.argmax(probs_full, dim=-1)

#                 probs_base = (100.0 * degra_features @ text_features_base.T).softmax(dim=-1)
#                 preds_base = torch.argmax(probs_base, dim=-1)

#             for pred_f, pred_b in zip(preds_full, preds_base):
#                 total += 1
#                 class_total[class_name] += 1
#                 if pred_f.item() == class_index:
#                     correct += 1
#                     class_correct[class_name] += 1

#                 true_base = base_class_map[class_name]
#                 pred_base_name = base_classes[pred_b.item()]
#                 base_total[true_base] += 1
#                 if pred_base_name == true_base:
#                     base_correct[true_base] += 1

#             batch_images.clear()

#     # 計算 accuracy
#     full_class_acc = [100.0 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0 for c in classes]
#     base_class_acc = [100.0 * base_correct[b] / base_total[b] if base_total[b] > 0 else 0.0 for b in base_classes]

#     return base_class_acc, full_class_acc

# checkpoints = {
#     "wild DACLIP pre-trained": "/mnt/hdd5/yicheng/daclip-uir/weights/wild-daclip_ViT-L-14.pt",
#     "Original CLIP": "/mnt/hdd5/yicheng/daclip-uir/da-clip/src/logs/daclip_ViT-B-32-start_epoch(original_clip)/checkpoints/epoch_1.pt",
#     "Our method": "/mnt/hdd5/yicheng/daclip-uir/da-clip/src/logs/daclip_ViT-B-32-20250421235346/checkpoints/epoch_187.pt",
# }


# all_base_results = {}
# all_full_results = {}

# for name, path in checkpoints.items():
#     print(f"\nEvaluating {name}")
#     base_acc, full_acc = evaluate_checkpoint(path)
#     all_base_results[name] = base_acc
#     all_full_results[name] = full_acc

# plt.figure(figsize=(10, 6))
# for name, accs in all_base_results.items():
#     plt.plot(base_classes, accs, marker='o', label=name)


# df_base = pd.DataFrame(all_base_results, index=base_classes)
# df_base.index.name = "Base Class"

# plt.figure(figsize=(8, 2 + 0.5 * len(base_classes)))
# plt.axis('off')
# table = plt.table(cellText=df_base.round(2).values,
#                   rowLabels=df_base.index,
#                   colLabels=df_base.columns,
#                   cellLoc='center',
#                   loc='center')
# table.scale(1, 1.5)
# plt.title("Base Class Accuracy Table")
# plt.savefig("base_class_accuracy_table.png", bbox_inches='tight')
# plt.show()



# plt.figure(figsize=(14, 6))
# for name, accs in all_full_results.items():
#     plt.plot(classes, accs, marker='o', label=name)

# plt.xticks(rotation=90)
# plt.title("Full Class Accuracy Comparison Across Checkpoints")
# plt.xlabel("Full Class (with strength)")
# plt.ylabel("Accuracy (%)")
# plt.ylim(0, 100)
# plt.legend(title="Checkpoint")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("compare_full_class_accuracy.png")
# plt.show()

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

# === 設定參數 ===
dataset_path = "/mnt/hdd5/yicheng/daclip-uir/universal-image-restoration/datasets/lsdir/test_OOD"
classes = [c for c in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, c))]
classes.sort()
print(f"Classes: {classes}")
batch_size = 32

base_class_map = {c: re.match(r'[a-zA-Z]+', c).group() for c in classes}
base_classes = sorted(set(base_class_map.values()))
print(f"Base Classes: {base_classes}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_checkpoint(checkpoint_path, model_name='daclip_ViT-B-32'):
    if checkpoint_path.endswith("wild-daclip_ViT-L-14.pt"):
        model, preprocess = open_clip.create_model_from_pretrained("daclip_ViT-L-14", pretrained=checkpoint_path)
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
    else:
        model, preprocess = open_clip.create_model_from_pretrained(model_name, pretrained=checkpoint_path)
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to(device)

    text_full = tokenizer(classes).to(device)
    text_base = tokenizer(base_classes).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features_full = model.encode_text(text_full)
        text_features_full /= text_features_full.norm(dim=-1, keepdim=True)

        text_features_base = model.encode_text(text_base)
        text_features_base /= text_features_base.norm(dim=-1, keepdim=True)

    total = 0
    correct = 0
    class_correct = {c: 0 for c in classes}
    class_total = {c: 0 for c in classes}
    base_correct = defaultdict(int)
    base_total = defaultdict(int)

    pred_logs = []  # 儲存 (filename, gt, pred)
    gt_labels = []
    pred_labels = []

    for class_index, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name, "LQ")
        batch_images = []
        batch_filenames = []

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
                        probs_full = (100.0 * degra_features @ text_features_full.T).softmax(dim=-1)
                        preds_full = torch.argmax(probs_full, dim=-1)

                    for fname, pred_f in zip(batch_filenames, preds_full):
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

                    batch_images.clear()
                    batch_filenames.clear()
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        # process remaining images
        if batch_images:
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                _, degra_features = model.encode_image(batch_tensor, control=True)
                degra_features /= degra_features.norm(dim=-1, keepdim=True)
                probs_full = (100.0 * degra_features @ text_features_full.T).softmax(dim=-1)
                preds_full = torch.argmax(probs_full, dim=-1)

            for fname, pred_f in zip(batch_filenames, preds_full):
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

            batch_images.clear()
            batch_filenames.clear()

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

    # 回傳準確率
    full_class_acc = [100.0 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0 for c in classes]
    base_class_acc = [100.0 * base_correct[b] / base_total[b] if base_total[b] > 0 else 0.0 for b in base_classes]
    return base_class_acc, full_class_acc

# === 執行評估 ===
checkpoints = {
    "wild DACLIP pre-trained": "/mnt/hdd5/yicheng/daclip-uir/weights/wild-daclip_ViT-L-14.pt",
    "Original CLIP": "/mnt/hdd5/yicheng/daclip-uir/da-clip/src/logs/daclip_ViT-B-32-start_epoch(original_clip)/checkpoints/epoch_1.pt",
    "Our method": "/mnt/hdd5/yicheng/daclip-uir/da-clip/src/logs/daclip_ViT-B-32-contrastive_learning_degraded_emb_2_degraded_emb/checkpoints/epoch_185.pt",
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
