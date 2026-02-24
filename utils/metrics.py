import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

def calculate_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(pred, target):
    return ssim(pred, target, data_range=1.0, size_average=True)
