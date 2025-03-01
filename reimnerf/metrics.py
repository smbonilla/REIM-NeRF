import torch
from kornia.losses import ssim as dssim
import lpips 

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    valid_mask = ~torch.isnan(image_gt)
        # check if there is ground truth of at least one pixel
    if torch.sum(valid_mask)==0:
        # print('NO depth samples found but depth loss is used')
        return torch.tensor(0.0)
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim(image_pred, image_gt, 3, reduction) # dissimilarity in [0, 1]
    return 1-2*dssim_ # in [-1, 1]

def calc_lpips(image_pred, image_gt, lpips_model=None):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    if lpips_model is None:
        lpips_ = lpips.LPIPS(net='vgg').to(image_pred.device)
    else:
        lpips_ = lpips_model.to(image_pred.device)
    lpips_val = lpips_(image_pred, image_gt).mean().double()
    return lpips_val  