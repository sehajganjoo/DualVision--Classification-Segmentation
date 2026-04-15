"""
segmentation/model.py
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMG_SIZE = 320

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# Helpers
def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=6,  dilation=6,  bias=False),
                                 nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
                                 nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
                                 nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.global_avg = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.5))

    def forward(self, x):
        h, w = x.shape[2:]
        x4   = F.interpolate(self.global_avg(x), size=(h, w), mode='bilinear', align_corners=False)
        return self.project(torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x), x4], dim=1))


#used during training
class SegmentationBackbone(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        resnet      = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool   = resnet.maxpool
        self.layer1 = resnet.layer1  
        self.layer2 = resnet.layer2   
        self.layer3 = resnet.layer3   
        self.layer4 = resnet.layer4
        self.aspp   = ASPP(2048, 256)
        self.up1    = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec1   = conv_block(256 + 1024, 256)
        self.up2    = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2   = conv_block(128 + 512,  128)
        self.up3    = nn.ConvTranspose2d(128, 64,  2, stride=2)
        self.dec3   = conv_block(64  + 256,  64)
        self.up4    = nn.ConvTranspose2d(64,  64,  2, stride=2)
        self.dec4   = conv_block(64  + 64,   64)
        self.final  = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        orig_size = x.shape[2:]
        x0 = self.layer0(x)
        xp = self.pool(x0)
        x1 = self.layer1(xp)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        z  = self.aspp(x4)
        z  = self.dec1(torch.cat([self.up1(z), x3], dim=1))
        z  = self.dec2(torch.cat([self.up2(z), x2], dim=1))
        z  = self.dec3(torch.cat([self.up3(z), x1], dim=1))
        z  = self.dec4(torch.cat([self.up4(z), x0], dim=1))
        z  = F.interpolate(z, size=orig_size, mode='bilinear', align_corners=False)
        return self.final(z)


# submission API class -------------------------------------------
class SegmentationModel:
    def __init__(self, weights_dir: str):
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone = SegmentationBackbone(num_classes=21).to(self.device)
        state          = torch.load(os.path.join(weights_dir, 'best_seg_model_A.pth'),
                                    map_location=self.device)
        self.backbone.load_state_dict(state)
        self.backbone.eval()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        t = val_transform(image=image)['image']
        return t.unsqueeze(0).float().to(self.device)

    def predict(self, image: np.ndarray) -> np.ndarray:
        orig_h, orig_w = image.shape[:2]
        with torch.no_grad():
            logits1 = self.backbone(self.preprocess(image))
            logits2 = torch.flip(self.backbone(self.preprocess(np.fliplr(image).copy())), dims=[-1])
        mask = torch.argmax((logits1 + logits2) / 2, dim=1)[0].cpu().numpy().astype(np.uint8)
        return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
