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


dataset_path = "/mnt/hdd7/yicheng/daclip-uir/universal-image-restoration/datasets_1/train/val"
classes = [c for c in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, c))]
classes.sort()
print(f"Classes: {classes}")
batch_size = 32

# base class 映射 (例如 blur10 -> blur)
base_class_map = {c: re.sub(r'\d+$', '', c) for c in classes}
base_classes = sorted(set(base_class_map.values()))
print(f"Base Classes: {base_classes}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_checkpoint(checkpoint_path, model_name='daclip_ViT-B-32'):

    if checkpoint_path == "/mnt/hdd7/yicheng/daclip-uir/weights/wild-daclip_ViT-L-14.pt":
        model, preprocess = open_clip.create_model_from_pretrained("daclip_ViT-L-14", pretrained=checkpoint_path)
        model.eval()
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
        model.to(device)
    else:
        model, preprocess = open_clip.create_model_from_pretrained(model_name, pretrained=checkpoint_path)
        model.eval()
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        model.to(device)

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

    for class_index, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name, "LQ")
        batch_images = []

        filenames = os.listdir(class_path)
        for filename in filenames:
            image_path = os.path.join(class_path, filename)
            try:
                image = preprocess(Image.open(image_path).convert("RGB"))
                batch_images.append(image)

                if len(batch_images) >= batch_size:
                    batch_tensor = torch.stack(batch_images).to(device)
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        _, degra_features = model.encode_image(batch_tensor, control=True)
                        degra_features /= degra_features.norm(dim=-1, keepdim=True)

                        probs_full = (100.0 * degra_features @ text_features_full.T).softmax(dim=-1)
                        preds_full = torch.argmax(probs_full, dim=-1)

                        probs_base = (100.0 * degra_features @ text_features_base.T).softmax(dim=-1)
                        preds_base = torch.argmax(probs_base, dim=-1)

                    for pred_f, pred_b in zip(preds_full, preds_base):
                        total += 1
                        class_total[class_name] += 1
                        if pred_f.item() == class_index:
                            correct += 1
                            class_correct[class_name] += 1

                        true_base = base_class_map[class_name]
                        pred_base_name = base_classes[pred_b.item()]
                        base_total[true_base] += 1
                        if pred_base_name == true_base:
                            base_correct[true_base] += 1

                    batch_images.clear()
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        if batch_images:
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                _, degra_features = model.encode_image(batch_tensor, control=True)
                degra_features /= degra_features.norm(dim=-1, keepdim=True)

                probs_full = (100.0 * degra_features @ text_features_full.T).softmax(dim=-1)
                preds_full = torch.argmax(probs_full, dim=-1)

                probs_base = (100.0 * degra_features @ text_features_base.T).softmax(dim=-1)
                preds_base = torch.argmax(probs_base, dim=-1)

            for pred_f, pred_b in zip(preds_full, preds_base):
                total += 1
                class_total[class_name] += 1
                if pred_f.item() == class_index:
                    correct += 1
                    class_correct[class_name] += 1

                true_base = base_class_map[class_name]
                pred_base_name = base_classes[pred_b.item()]
                base_total[true_base] += 1
                if pred_base_name == true_base:
                    base_correct[true_base] += 1

            batch_images.clear()

    # 計算 accuracy
    full_class_acc = [100.0 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0 for c in classes]
    base_class_acc = [100.0 * base_correct[b] / base_total[b] if base_total[b] > 0 else 0.0 for b in base_classes]

    return base_class_acc, full_class_acc

checkpoints = {
    # "wild DACLIP pre-trained": "/mnt/hdd7/yicheng/daclip-uir/weights/wild-daclip_ViT-L-14.pt",
    "Original CLIP": "/mnt/hdd7/yicheng/daclip-uir/da-clip/src/logs/daclip_ViT-B-32-start_epoch/checkpoints/epoch_1.pt",
    "Our method": "/mnt/hdd7/yicheng/daclip-uir/da-clip/src/logs/daclip_ViT-B-32-20250328165905/checkpoints/epoch_102.pt",
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

# plt.title("Base Class Accuracy Comparison Across Checkpoints")
# plt.xlabel("Base Class")
# plt.ylabel("Accuracy (%)")
# plt.ylim(0, 100)
# plt.legend(title="Checkpoint")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("compare_base_class_accuracy.png")
# plt.show()

df_base = pd.DataFrame(all_base_results, index=base_classes)
df_base.index.name = "Base Class"
# 顯示作為表格圖像（可插入報告）
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




















# # 設定參數
# dataset_path = "/mnt/hdd7/yicheng/daclip-uir/universal-image-restoration/datasets_1/train/val"
# classes = [c for c in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, c))]
# classes.sort()
# print(f"Classes: {classes}")
# batch_size = 64

# # base class 映射 (例如 blur10 -> blur)
# base_class_map = {c: re.sub(r'\d+$', '', c) for c in classes}
# base_classes = sorted(set(base_class_map.values()))
# print(f"Base Classes: {base_classes}")

# # 載入模型
# checkpoint = '/mnt/hdd7/yicheng/daclip-uir/da-clip/src/logs/daclip_ViT-B-32-20250328102353/checkpoints/epoch_80.pt'
# model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=checkpoint)
# model.eval()
# tokenizer = open_clip.get_tokenizer('ViT-B-32')

# # 設定裝置
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)


# # text_full_class.sort()
# # print(text_full_class)
# # Tokenize 完整類別和 base 類別
# text_full = tokenizer(classes).to(device)
# text_base = tokenizer(base_classes).to(device)

# with torch.no_grad(), torch.cuda.amp.autocast():
#     text_features_full = model.encode_text(text_full)
#     text_features_full /= text_features_full.norm(dim=-1, keepdim=True)

#     text_features_base = model.encode_text(text_base)
#     text_features_base /= text_features_base.norm(dim=-1, keepdim=True)

# # 統計資料
# total = 0
# correct = 0
# class_correct = {c: 0 for c in classes}
# class_total = {c: 0 for c in classes}

# base_correct = defaultdict(int)
# base_total = defaultdict(int)

# # 分類預測開始
# for class_index, class_name in enumerate(classes):
#     print(f"\nProcessing {class_name}...")
#     class_path = os.path.join(dataset_path, class_name, "LQ")

#     batch_images = []
#     for filename in tqdm(os.listdir(class_path)):
#         image_path = os.path.join(class_path, filename)
#         try:
#             image = preprocess(Image.open(image_path).convert("RGB"))
#             batch_images.append(image)

#             if len(batch_images) >= batch_size:
#                 batch_tensor = torch.stack(batch_images).to(device)
#                 with torch.no_grad(), torch.cuda.amp.autocast():
#                     _, degra_features = model.encode_image(batch_tensor, control=True)
#                     degra_features /= degra_features.norm(dim=-1, keepdim=True)

#                     probs_full = (100.0 * degra_features @ text_features_full.T).softmax(dim=-1)
#                     preds_full = torch.argmax(probs_full, dim=-1)

#                     probs_base = (100.0 * degra_features @ text_features_base.T).softmax(dim=-1)
#                     preds_base = torch.argmax(probs_base, dim=-1)

#                 for pred_f, pred_b in zip(preds_full, preds_base):
#                     total += 1
#                     class_total[class_name] += 1
#                     if pred_f.item() == class_index:
#                         correct += 1
#                         class_correct[class_name] += 1

#                     true_base = base_class_map[class_name]
#                     pred_base_name = base_classes[pred_b.item()]
#                     base_total[true_base] += 1
#                     if pred_base_name == true_base:
#                         base_correct[true_base] += 1

#                 batch_images.clear()

#         except Exception as e:
#             print(f"Error processing {image_path}: {e}")

#     # 處理剩下的 batch
#     if batch_images:
#         batch_tensor = torch.stack(batch_images).to(device)
#         with torch.no_grad(), torch.cuda.amp.autocast():
#             _, degra_features = model.encode_image(batch_tensor, control=True)
#             degra_features /= degra_features.norm(dim=-1, keepdim=True)

#             probs_full = (100.0 * degra_features @ text_features_full.T).softmax(dim=-1)
#             preds_full = torch.argmax(probs_full, dim=-1)

#             probs_base = (100.0 * degra_features @ text_features_base.T).softmax(dim=-1)
#             preds_base = torch.argmax(probs_base, dim=-1)

#         for pred_f, pred_b in zip(preds_full, preds_base):
#             total += 1
#             class_total[class_name] += 1
#             if pred_f.item() == class_index:
#                 correct += 1
#                 class_correct[class_name] += 1

#             true_base = base_class_map[class_name]
#             pred_base_name = base_classes[pred_b.item()]
#             base_total[true_base] += 1
#             if pred_base_name == true_base:
#                 base_correct[true_base] += 1

#         batch_images.clear()

# # 顯示分類準確率
# print("\n=== Full Class Accuracy ===")
# full_class_acc = []
# for c in classes:
#     acc = 100.0 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
#     full_class_acc.append(acc)
#     print(f"{c}: {acc:.2f}% ({class_correct[c]}/{class_total[c]})")

# print("\n=== Base Class Accuracy (Ignoring Strength) ===")
# base_class_acc = []
# for b in base_classes:
#     acc = 100.0 * base_correct[b] / base_total[b] if base_total[b] > 0 else 0.0
#     base_class_acc.append(acc)
#     print(f"{b}: {acc:.2f}% ({base_correct[b]}/{base_total[b]})")

# # 總體準確率
# print(f"\nTotal: {total}, Correct: {correct}")
# print(f"Overall Accuracy (Full Class): {100.0 * correct / total:.2f}%")


# # === 1. Grouped Line Plot (Accuracy Trend per Strength) ===
# grouped_data = defaultdict(list)
# for c, acc in zip(classes, full_class_acc):
#     base = base_class_map[c]
#     grouped_data[base].append((c, acc))

# plt.figure(figsize=(14, 6))
# for base, items in grouped_data.items():
#     items.sort()
#     labels = [k for k, _ in items]
#     values = [v for _, v in items]
#     plt.plot(labels, values, marker='o', label=base)

# plt.xticks(rotation=90)
# plt.ylabel('Accuracy (%)')
# plt.title('Accuracy Trend Across Degradation Strengths')
# plt.legend()
# plt.tight_layout()
# plt.savefig('trend_grouped_plot.png')
# plt.close()

# plt.figure(figsize=(8, 5))
# plt.plot(base_classes, base_class_acc, marker='o', linestyle='-', linewidth=2)
# plt.title("Base Class Accuracy (Ignoring Strength)")
# plt.xlabel("Base Class")
# plt.ylabel("Accuracy (%)")
# plt.ylim(0, 100)
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('Base Class Accuracy trend_grouped_plot.png')
# plt.close()


# text_full_class = [
#     "blur1.0", "blur1.5", "blur2.0","blur2.5", 
#     "blur3.0", "blur3.5","blur4.0","blur0.5", 
#     "jpeg10","jpeg20","jpeg30","jpeg40",
#     "jpeg50","jpeg60","jpeg70","jpeg80",
#     "noisy10","noisy15","noisy20","noisy25",
#     "noisy30","noisy35","noisy40","noisy5",
#     "resize1.0","resize1.5","resize2.0","resize2.5",
#     "resize3.0","resize3.5","resize4.0","resize0.5"
# ]