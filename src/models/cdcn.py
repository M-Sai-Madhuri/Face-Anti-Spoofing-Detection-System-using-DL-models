"""
CDCN - Central Difference Convolutional Network
Captures micro-texture patterns for face anti-spoofing detection.
Based on: "Central Difference Convolutional Networks" (CVPR 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CentralDifferenceConv2d(nn.Module):
    """
    Central Difference Convolution (CDC) layer.
    Combines standard convolution with gradient-based information.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, bias=False, theta=0.7):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding, bias=bias)
        self.theta = theta
    
    def forward(self, x):
        # Standard convolution
        out_normal = self.conv(x)
        
        if abs(self.theta) < 1e-8:
            return out_normal
        
        # Central difference convolution
        kernel = self.conv.weight
        kernel_diff = kernel.sum(dim=[2, 3], keepdim=True)
        out_diff = F.conv2d(x, kernel_diff, self.conv.bias, 
                           self.conv.stride, padding=0)
        
        return out_normal - self.theta * out_diff


class CDCBlock(nn.Module):
    """CDC Block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels, out_channels, theta=0.7):
        super().__init__()
        self.cdc = CentralDifferenceConv2d(in_channels, out_channels, 
                                           kernel_size=3, padding=1, theta=theta)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.cdc(x)))


class CDCN(nn.Module):
    """
    Central Difference Convolutional Network for Face Anti-Spoofing.
    
    Architecture:
    - Multi-scale CDC feature extraction
    - Lightweight design (~1.5M parameters)
    - Captures fine-grained texture differences
    """
    
    def __init__(self, num_classes=2, pretrained=False, theta=0.7, dropout=0.5):
        super().__init__()
        
        # pretrained is ignored for CDCN (no pretrained weights available)
        self.theta = theta
        
        # Stem
        self.stem = nn.Sequential(
            CDCBlock(3, 64, theta),
            CDCBlock(64, 128, theta),
            nn.MaxPool2d(2, 2),
        )
        
        # Stage 1
        self.stage1 = nn.Sequential(
            CDCBlock(128, 128, theta),
            CDCBlock(128, 196, theta),
            nn.MaxPool2d(2, 2),
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            CDCBlock(196, 128, theta),
            CDCBlock(128, 128, theta),
            nn.MaxPool2d(2, 2),
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            CDCBlock(128, 128, theta),
            CDCBlock(128, 64, theta),
            nn.MaxPool2d(2, 2),
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(32, num_classes),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_features(self, x):
        """Extract features before classifier."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        return x.view(x.size(0), -1)


if __name__ == "__main__":
    # Test the model
    model = CDCN(num_classes=2)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"CDCN Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
