import torch
from torch.utils.data import DataLoader
from models.joint_model import JointModel
from datasets.llie_dataset import LLIE_Dataset
from utils.losses import VGGPerceptualLoss, combined_loss
from utils.metrics import calculate_psnr
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = LLIE_Dataset("dataset/train/low", "dataset/train/high")
val_dataset   = LLIE_Dataset("dataset/val/low", "dataset/val/high")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)

model = JointModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
perceptual = VGGPerceptualLoss().to(device)

best_psnr = 0

for epoch in range(20):
    model.train()

    for low, high in train_loader:
        low, high = low.to(device), high.to(device)

        output = model(low)
        loss = combined_loss(output, high, perceptual)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch", epoch+1)

    model.eval()
    total_psnr = 0

    with torch.no_grad():
        for low, high in val_loader:
            low, high = low.to(device), high.to(device)
            output = model(low)
            total_psnr += calculate_psnr(output, high).item()

    avg_psnr = total_psnr / len(val_loader)
    print("Validation PSNR:", avg_psnr)

    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print("Best model saved")
