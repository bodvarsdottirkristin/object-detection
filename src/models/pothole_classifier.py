import torch
import torch.nn as nn
from torchvision import models

from src.utils.logger import get_logger


logger = get_logger(__name__)


class PotholeClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False, dropout_p=0.5):
        super(PotholeClassifier, self).__init__()
        
        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            logger.info("Loaded ResNet18 with pretrained weights.")
        else:
            self.backbone = models.resnet18(weights=None)
            logger.info("Loaded ResNet18 without pretrained weights.")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
            logger.info("Backbone frozen except final layer.")

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = PotholeClassifier(num_classes=2)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)

    logger.info(f"Model output shape: {output.shape}")  # Should be [4, 2] for batch size 4 and 2 classes
    logger.info("PotholeClassifier model test completed successfully.")