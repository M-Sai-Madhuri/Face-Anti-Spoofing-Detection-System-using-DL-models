"""
Frequency-CNN - DCT-based Frequency Domain Analysis
Detects print/screen artifacts through frequency analysis.
Based on: Frequency domain analysis for face anti-spoofing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


class DCTLayer(nn.Module):
    """
    Discrete Cosine Transform layer.
    Converts spatial domain to frequency domain.
    """
    
    def __init__(self, size=224):
        super().__init__()
        self.size = size
        
        # Precompute DCT basis matrix
        dct_matrix = self._create_dct_matrix(size)
        self.register_buffer('dct_matrix', dct_matrix)
    
    def _create_dct_matrix(self, n):
        """Create DCT-II matrix."""
        dct = torch.zeros(n, n)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    dct[k, i] = 1 / np.sqrt(n)
                else:
                    dct[k, i] = np.sqrt(2 / n) * np.cos(np.pi * k * (2 * i + 1) / (2 * n))
        return dct
    
    def forward(self, x):
        """Apply 2D DCT to each channel."""
        b, c, h, w = x.shape
        
        # Resize if needed
        if h != self.size or w != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False)
        
        # Apply DCT: D @ X @ D.T
        x = torch.matmul(self.dct_matrix, x)
        x = torch.matmul(x, self.dct_matrix.T)
        
        return x


class FrequencyBlock(nn.Module):
    """Convolutional block for frequency features."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class FrequencyCNN(nn.Module):
    """
    Frequency Domain CNN for Face Anti-Spoofing.
    
    Architecture:
    - DCT transform for frequency features
    - Multi-scale frequency analysis
    - Lightweight CNN classifier
    - ~1.8M parameters
    """
    
    def __init__(self, num_classes=2, pretrained=False, dropout=0.6, dct_size=112, version='v2'):
        super().__init__()
        
        self.version = version
        # pretrained is ignored for FrequencyCNN (no pretrained weights available)
        # DCT Transform (smaller size for efficiency)
        self.dct = DCTLayer(size=dct_size)
        self.dct_size = dct_size
        
        # Frequency feature extractor
        self.freq_encoder = nn.Sequential(
            FrequencyBlock(3, 32),      # 112 -> 56
            FrequencyBlock(32, 64),     # 56 -> 28
            FrequencyBlock(64, 128),    # 28 -> 14
            FrequencyBlock(128, 256),   # 14 -> 7
        )
        
        if version == 'v1':
            # Original Spatial feature extractor (parallel branch)
            self.spatial_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(7),
            )
            self.spatial_pool = nn.Identity()
            spatial_out_channels = 128
        else:
            # Spatial feature extractor (parallel branch) - Using pre-trained backbone (v2)
            backbone = models.mobilenet_v3_small(pretrained=True)
            self.spatial_encoder = backbone.features
            # MobileNetV3-Small features output 576 channels for 224x224 input
            # We'll use AdaptiveAvgPool2d to ensure it matches the frequency branch (7x7)
            self.spatial_pool = nn.AdaptiveAvgPool2d(7)
            spatial_out_channels = 576
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(256 + spatial_out_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4 if version == 'v2' else dropout / 2),
            nn.Linear(128, num_classes),
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Resize for DCT
        x_dct = F.interpolate(x, size=(self.dct_size, self.dct_size), 
                              mode='bilinear', align_corners=False)
        
        # Frequency branch
        freq_features = self.dct(x_dct)
        freq_features = self.freq_encoder(freq_features)
        
        # Spatial branch
        spatial_features = self.spatial_encoder(x)
        spatial_features = self.spatial_pool(spatial_features)
        
        # Fusion
        combined = torch.cat([freq_features, spatial_features], dim=1)
        fused = self.fusion(combined)
        
        # Classification
        x = self.gap(fused)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def get_features(self, x):
        """Extract features before classifier."""
        x_dct = F.interpolate(x, size=(self.dct_size, self.dct_size), 
                              mode='bilinear', align_corners=False)
        freq_features = self.dct(x_dct)
        freq_features = self.freq_encoder(freq_features)
        spatial_features = self.spatial_encoder(x)
        spatial_features = self.spatial_pool(spatial_features)
        combined = torch.cat([freq_features, spatial_features], dim=1)
        fused = self.fusion(combined)
        x = self.gap(fused)
        return x.view(x.size(0), -1)


if __name__ == "__main__":
    # Test the model
    model = FrequencyCNN(num_classes=2)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"FrequencyCNN Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
