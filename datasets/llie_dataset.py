import os
import glob
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class LLIE_Dataset(Dataset):
    def __init__(self, low_dir, high_dir, size=256):
        self.low_images = sorted(glob.glob(os.path.join(low_dir, "*")))
        self.high_images = sorted(glob.glob(os.path.join(high_dir, "*")))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size))
        ])

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low = cv2.imread(self.low_images[idx])
        high = cv2.imread(self.high_images[idx])

        low = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)
        high = cv2.cvtColor(high, cv2.COLOR_BGR2RGB)

        low = self.transform(low)
        high = self.transform(high)

        return low, high
