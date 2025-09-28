# -*- coding: utf-8 -*-
#
# This script provides implementations for the loss functions described in the VideoGigaGAN paper.
# It includes Charbonnier loss, a GAN loss wrapper, R1 regularization, and a wrapper for LPIPS.

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import lpips
except ImportError:
    print("Please install the lpips library: pip install lpips")

class CharbonnierLoss(nn.Module):
    """Charbonnier loss (a variant of L1 loss)."""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss

class GANLoss(nn.Module):
    """Wrapper for GAN loss."""
    def __init__(self, gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.register_buffer('real_label', torch.tensor(real_label_val))
        self.register_buffer('fake_label', torch.tensor(fake_label_val))
        if gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError(f'GAN type {gan_type} is not implemented.')

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target = self.real_label.expand_as(prediction)
        else:
            target = self.fake_label.expand_as(prediction)
        return target

    def forward(self, prediction, target_is_real, is_disc=False):
        if self.gan_type == 'wgangp':
            if is_disc:
                if target_is_real:
                    return -prediction.mean()
                else:
                    return prediction.mean()
            else: # for generator
                return -prediction.mean()
        
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

def r1_regularization(real_pred, real_img):
    """
    R1 regularization: gradient penalty on real data.
    real_pred: output of discriminator for real samples (scalar per sample)
    real_img: real input images (requires_grad=True)
    """
    grad_real = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_img,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Use .reshape instead of .view to avoid contiguous error
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

class LPIPSLoss(nn.Module):
    """Wrapper for LPIPS loss."""
    def __init__(self, net='alex'):
        super(LPIPSLoss, self).__init__()
        self.model = lpips.LPIPS(net=net)

    def forward(self, x, y):
        # LPIPS expects images in range [-1, 1]
        return self.model(x * 2 - 1, y * 2 - 1).mean()
