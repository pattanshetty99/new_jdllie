import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from pytorch_msssim import ssim

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16]
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        return F.l1_loss(self.vgg(pred), self.vgg(target))

def combined_loss(pred, target, perceptual):
    l1 = F.l1_loss(pred, target)
    ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
    perc = perceptual(pred, target)
    return l1 + 0.2*ssim_loss + 0.1*perc
