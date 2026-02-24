import os
import torch
from torch.utils.data import DataLoader
from models.joint_model import JointModel
from datasets.llie_dataset import LLIE_Dataset
from utils.metrics import calculate_psnr, calculate_ssim

# ----------------------
# Settings
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "checkpoints/best_model.pth"

test_low_dir = "dataset/test/low"
test_high_dir = "dataset/test/high"

# ----------------------
# Load Dataset
# ----------------------
test_dataset = LLIE_Dataset(test_low_dir, test_high_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ----------------------
# Load Model
# ----------------------
model = JointModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ----------------------
# Evaluation
# ----------------------
total_psnr = 0
total_ssim = 0
total_images = 0

with torch.no_grad():
    for low, high in test_loader:
        low = low.to(device)
        high = high.to(device)

        output = model(low)

        psnr_val = calculate_psnr(output, high)
        ssim_val = calculate_ssim(output, high)

        total_psnr += psnr_val.item()
        total_ssim += ssim_val.item()
        total_images += 1

avg_psnr = total_psnr / total_images
avg_ssim = total_ssim / total_images

print("\n===== Evaluation Results =====")
print(f"Average PSNR: {avg_psnr:.2f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")
print("==============================")
