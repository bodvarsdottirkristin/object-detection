import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PotholeClassifier(nn.Module):
    """
    Transfer learning classifier for pothole vs background proposals.
    Uses ResNet50 backbone for binary classification.
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        """
        Initialize classifier
        
        Args:
            num_classes (int): Number of output classes (2 for binary)
            pretrained (bool): Use ImageNet pretrained weights
        """
        super(PotholeClassifier, self).__init__()
        
        # Load pretrained ResNet18
        backbone = models.resnet50(pretrained=pretrained)
        
        # Replace final layer for binary classification
        num_features = backbone.fc.in_features
        backbone.fc = nn.Linear(num_features, num_classes)
        
        self.model = backbone
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 3, H, W)
            
        Returns:
            logits: Output logits (batch_size, num_classes)
        """
        return self.model(x)


# Example usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create model
    model = PotholeClassifier(num_classes=2, pretrained=True).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*60)
    print("POTHOLE CLASSIFIER (ResNet18)")
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\nModel architecture:")
    print(model)
    print("="*60 + "\n")
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(x)
        probs = F.softmax(output, dim=1)
        predictions = torch.argmax(probs, dim=1)
        
        print("Forward pass test:")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Predictions: {predictions.tolist()}")
        print(f"  Confidence: {probs.max(dim=1).values.tolist()}")
    
    print("\n✓ Model ready for training")