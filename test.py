import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from models.joint_model import JointModel

# ----------------------
# Settings
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "checkpoints/best_model.pth"
input_dir = "dataset/test/low"
output_dir = "results"

os.makedirs(output_dir, exist_ok=True)

# ----------------------
# Load Model
# ----------------------
model = JointModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.ToTensor()

# ----------------------
# Testing Loop
# ----------------------
with torch.no_grad():
    for img_name in os.listdir(input_dir):

        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img).unsqueeze(0).to(device)

        output = model(img_tensor)

        output = output.squeeze().cpu().permute(1, 2, 0).numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

        print(f"Saved: {img_name}")

print("Testing completed.")
