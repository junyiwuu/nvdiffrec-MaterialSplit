# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import pandas as pd
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

#----------------------------------------------------------------------------
# HDR image losses
#----------------------------------------------------------------------------


_vgg_conv1_2 = models.vgg19(pretrained=True).features[:4].eval().to(device='cuda').eval()
for p in _vgg_conv1_2.parameters():
        p.requires_grad = False

_mean = torch.tensor([0.485, 0.456, 0.406], device='cuda')[:, None, None] # [3,1,1]
_std  = torch.tensor([0.229, 0.224, 0.225], device='cuda')[:, None, None]


def _tonemap_srgb(f):  # gamma correction
    return torch.where(f > 0.0031308, torch.pow(torch.clamp(f, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*f)

# symmetric mean absolute percentage error
def _SMAPE(img, target, eps=0.01):
    nom = torch.abs(img - target)
    denom = torch.abs(img) + torch.abs(target) + 0.01
    return torch.mean(nom / denom)

# relative mean squared error (author made)
def _RELMSE(img, target, eps=0.1):
    nom = (img - target) * (img - target)
    denom = img * img + target * target + 0.1 
    return torch.mean(nom / denom)


def _perceptualLoss(img, target, eps=0.0001):
    # nvdiffrec image shape is [n, h, w, c], so need permute
    # 2d convolutions expects channels first tensorL [B, C , H , W]
    pred_img = (img.permute(0, 3, 1, 2) - _mean) / _std
    pred_target = (target.permute(0,3,1,2) - _mean) / _std

    feat_img = _vgg_conv1_2(pred_img)
    feat_target = _vgg_conv1_2(pred_target)

    loss = F.mse_loss(feat_img, feat_target)

    return loss

def _msssim(img, target, eps=0.0001):
    pred_img = img.permute(0, 3, 1, 2) 
    pred_target = target.permute(0,3,1,2) 
    return ms_ssim(pred_img, pred_target)



# L1_loss : Mean Absolute Error L1 = mean(|A-B|)
def image_loss_fn(img, target, loss, tonemapper):
    if tonemapper == 'log_srgb':
        img    = _tonemap_srgb(torch.log(torch.clamp(img, min=0, max=65535) + 1))
        target = _tonemap_srgb(torch.log(torch.clamp(target, min=0, max=65535) + 1))

    if loss == 'mse':
        return torch.nn.functional.mse_loss(img, target)
    elif loss == 'smape':
        return _SMAPE(img, target)
    elif loss == 'relmse':
        return _RELMSE(img, target)
    elif loss == 'perceptual':
        return _perceptualLoss(img, target)
    elif loss == "msssim":
        return _msssim(img, target)
    else:
        return torch.nn.functional.l1_loss(img, target)



# Adapted from: https://github.com/victoresque/pytorch-template/blob/master/utils/util.py
class MetricTracker:  #  writer没有设置过，只是用他的累积和平均的功能 
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
