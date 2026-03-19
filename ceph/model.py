import torch
import torch.nn as nn
from torchvision import models

class LandmarkModel(nn.Module):
    def __init__(self, num_landmarks=29, backbone='resnet50'):
        super().__init__()
        # Sử dụng pretrained ResNet backbone
        if backbone == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif backbone == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif backbone == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif backbone == 'resnet101':
            resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Lấy feature dimension từ fc layer cuối
        feat_dim = resnet.fc.in_features

        # Bỏ fc layer cuối, giữ lại phần feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.num_landmarks = num_landmarks

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_landmarks * 2)  # x, y cho 29 landmarks
        )

    def forward(self, x):
        features = self.backbone(x)
        features = features.flatten(1)  # (batch, feat_dim)
        output = self.head(features)
        return output.reshape(-1, self.num_landmarks, 2) 