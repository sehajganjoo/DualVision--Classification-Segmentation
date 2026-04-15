"""
classification/model.py
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


# Constants
CLASS_NAMES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

IMG_SIZE = 320

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# used during training
class ClassificationBackbone(nn.Module):
    """ResNet-50 + multi-scale pooling head for 20-class multilabel classification."""
    def __init__(self, num_classes=20):
        super().__init__()
        backbone     = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.gap      = nn.AdaptiveAvgPool2d(1)
        self.gmp      = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(1024, 512),      nn.BatchNorm1d(512),  nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(torch.cat([self.gap(feat).flatten(1),
                                          self.gmp(feat).flatten(1)], dim=1))


# submission API class-------------------------------------------
class ClassificationModel:
    def __init__(self, weights_dir: str):
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone  = ClassificationBackbone(num_classes=20).to(self.device)
        state           = torch.load(os.path.join(weights_dir, 'best_cls_model.pth'),
                                     map_location=self.device)
        self.backbone.load_state_dict(state)
        self.backbone.eval()

        thresh_path     = os.path.join(weights_dir, 'best_thresholds.npy')
        self.thresholds = np.load(thresh_path) if os.path.exists(thresh_path) else np.full(20, 0.5)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        t = val_transform(image=image)['image']
        return t.unsqueeze(0).float().to(self.device)

    def predict(self, image: np.ndarray) -> dict:
        with torch.no_grad():
            p1 = torch.sigmoid(self.backbone(self.preprocess(image)))
            p2 = torch.sigmoid(self.backbone(self.preprocess(np.fliplr(image).copy())))
        probs = ((p1 + p2) / 2).cpu().numpy()[0]
        return {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}

