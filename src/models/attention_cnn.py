"""
Attention-CNN - MobileNetV3 with CBAM Attention
Lightweight and efficient for mobile deployment.
Based on: MobileNetV3 + Convolutional Block Attention Module (CBAM)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ChannelAttention(nn.Module):
    """Channel Attention Module from CBAM."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial Attention Module from CBAM."""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        
        return x * out.expand_as(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AttentionCNN(nn.Module):
    """
    MobileNetV3-Small with CBAM attention for Face Anti-Spoofing.
    
    Architecture:
    - MobileNetV3-Small backbone (pretrained on ImageNet)
    - CBAM attention module
    - Custom classification head
    - ~2.5M parameters
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super().__init__()
        
        # Load MobileNetV3-Small backbone
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            backbone = models.mobilenet_v3_small(weights=weights)
        else:
            backbone = models.mobilenet_v3_small(weights=None)
        
        # Extract features (remove classifier)
        self.features = backbone.features
        
        # Get the output channels of the last conv layer
        # MobileNetV3-Small: 576 channels after features
        self.feature_channels = 576
        
        # CBAM Attention
        self.cbam = CBAM(self.feature_channels, reduction=16)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_channels, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        
        # Apply CBAM attention
        x = self.cbam(x)
        
        # Global pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        return x
    
    def get_features(self, x):
        """Extract features before classifier."""
        x = self.features(x)
        x = self.cbam(x)
        x = self.gap(x)
        return x.view(x.size(0), -1)


if __name__ == "__main__":
    # Test the model
    model = AttentionCNN(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"AttentionCNN Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
