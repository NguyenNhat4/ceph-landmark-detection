import torch
import torch.nn as nn
import timm

class LandmarkModel(nn.Module):
    def __init__(self, num_landmarks=29, backbone='efficientnet_b0'):
        super().__init__()
        # Sử dụng pretrained backbone
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)

        # Feature dimension
        feat_dim = self.backbone.num_features

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_landmarks * 2)  # x, y cho 29 landmarks
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output.reshape(-1, 29, 2) 