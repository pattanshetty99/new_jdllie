import torch

def rgb_to_ycbcr(img):
    R = img[:,0:1,:,:]
    G = img[:,1:2,:,:]
    B = img[:,2:3,:,:]

    Y  = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.168736*R - 0.331264*G + 0.5*B + 0.5
    Cr = 0.5*R - 0.418688*G - 0.081312*B + 0.5

    return torch.cat([Y, Cb, Cr], dim=1)

def ycbcr_to_rgb(img):
    Y  = img[:,0:1,:,:]
    Cb = img[:,1:2,:,:] - 0.5
    Cr = img[:,2:3,:,:] - 0.5

    R = Y + 1.402*Cr
    G = Y - 0.344136*Cb - 0.714136*Cr
    B = Y + 1.772*Cb

    return torch.cat([R,G,B], dim=1)
