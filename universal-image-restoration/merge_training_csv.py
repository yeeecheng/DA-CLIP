import os
import pandas as pd

root_dir = "/mnt/hdd7/yicheng/daclip-uir/universal-image-restoration/datasets_1/train/val"
output_csv = "/mnt/hdd7/yicheng/daclip-uir/universal-image-restoration/datasets_1/train/val/merged_daclip_train.csv"

# 列出所有 degradation 子資料夾
degradation_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
print(degradation_dirs)
# 初始化 DataFrame
merged_df = pd.DataFrame(columns=["filepath", "title"])

future_df = {"filepath":[], "title":[]}
# 遍歷每個 degradation 類型的資料夾，讀取 daclip_val.csv 並合併
for deg in degradation_dirs:
    csv_path = os.path.join(root_dir, deg, "daclip_val.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep='\t')
    
        future_df["filepath"].extend(df["filepath"].tolist())
        future_df["title"].extend(df["title"].tolist())
    else:
        print(f"Warning: {csv_path} does not exist.")



pd.DataFrame.from_dict(future_df).to_csv(output_csv, index=False, sep="\t")
print(f"Merged CSV saved to {output_csv}")
