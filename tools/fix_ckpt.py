import torch

# 输入和输出文件路径
input_ckpt_path = "/workspace/OpenPCDet/tools/inference/ckpt/VoxelNeXt_Argo2.pth"  # 你原始的权重文件路径
output_ckpt_path = "/workspace/OpenPCDet/tools/inference/ckpt/VoxelNeXt_Argo2_fixed.pth"  # 修正后要保存的路径

# 加载原始权重
raw_weights = torch.load(input_ckpt_path, map_location='cpu')

# 重新封装为OpenPCDet预期的格式
fixed_weights = {'model_state': raw_weights}

# 保存修正后的权重文件
torch.save(fixed_weights, output_ckpt_path)
print(f"权重文件已修正并保存至: {output_ckpt_path}")