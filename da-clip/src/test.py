import torch

ckpt_path = "/mnt/hdd5/yicheng/daclip-uir/da-clip/src/logs/daclip_ViT-B-32-20250424061755/checkpoints/epoch_190.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

print("🔍 Available Keys in Checkpoint:")
print(ckpt.keys())  # 常見有 'model', 'state_dict', 'module', etc.

state_dict = ckpt['state_dict']
for k in state_dict.keys():
    print(k)
