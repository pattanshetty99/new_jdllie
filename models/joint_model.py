import torch
import torch.nn as nn
from models.luma_net import LumaNet
from utils.color_utils import rgb_to_ycbcr, ycbcr_to_rgb

class JointModel(nn.Module):
    def __init__(self):
        super(JointModel, self).__init__()
        self.luma_net = LumaNet()

    def forward(self, x):
        ycbcr = rgb_to_ycbcr(x)

        Y  = ycbcr[:,0:1,:,:]
        Cb = ycbcr[:,1:2,:,:]
        Cr = ycbcr[:,2:3,:,:]

        Y_enhanced = self.luma_net(Y)

        merged = torch.cat([Y_enhanced, Cb, Cr], dim=1)
        rgb = ycbcr_to_rgb(merged)

        return torch.clamp(rgb, 0, 1)
