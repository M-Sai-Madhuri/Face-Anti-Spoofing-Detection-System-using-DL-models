"""
CNN-ViT Hybrid - ResNet18 + Vision Transformer
Combines local CNN features with global ViT attention.
Based on: Hybrid CNN-Transformer architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


class PatchEmbedding(nn.Module):
    """Convert CNN feature maps to patch embeddings for ViT."""
    
    def __init__(self, in_channels, embed_dim, patch_size=1):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = self.norm(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self Attention."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CNNViT(nn.Module):
    """
    Hybrid CNN-ViT for Face Anti-Spoofing.
    
    Architecture:
    - ResNet18 backbone for local features (pretrained)
    - Vision Transformer head for global context
    - Combines local texture with global patterns
    - ~5M parameters
    """
    
    def __init__(self, num_classes=2, pretrained=True, embed_dim=256, 
                 num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        # ResNet18 backbone (remove last layers)
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
        else:
            resnet = models.resnet18(weights=None)
        
        # Keep layers up to layer3 (output: 256 channels, 14x14 for 224 input)
        self.cnn_backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        
        # CNN output: 256 channels
        cnn_out_channels = 256
        
        # Patch embedding (1x1 patches from CNN features)
        self.patch_embed = PatchEmbedding(cnn_out_channels, embed_dim)
        
        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding (14x14 = 196 patches + 1 cls token)
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, dropout=dropout) 
              for _ in range(num_layers)]
        )
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # CNN feature extraction
        x = self.cnn_backbone(x)  # (B, 256, 14, 14)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, 196, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, embed_dim)
        
        # Add positional embedding
        if x.shape[1] != self.pos_embed.shape[1]:
            # Interpolate positional embeddings if sequence length mismatch (e.g. 112x112 input)
            pos_embed = self.pos_embed
            cls_pos = pos_embed[:, :1, :]
            patch_pos = pos_embed[:, 1:, :]
            
            # Reshape to grid
            N = patch_pos.shape[1]
            grid_size = int(math.sqrt(N))
            patch_pos = patch_pos.reshape(1, grid_size, grid_size, -1).permute(0, 3, 1, 2)
            
            # Interpolate
            current_N = x.shape[1] - 1
            current_grid = int(math.sqrt(current_N))
            patch_pos = F.interpolate(patch_pos, size=(current_grid, current_grid), 
                                      mode='bilinear', align_corners=False)
            
            # Flatten back
            patch_pos = patch_pos.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = torch.cat([cls_pos, patch_pos], dim=1)
            
            x = self.pos_drop(x + pos_embed)
        else:
            x = self.pos_drop(x + self.pos_embed)
        
        # Transformer
        x = self.transformer(x)
        
        # Classification (use cls token)
        x = self.norm(x[:, 0])
        x = self.head(x)
        
        return x
    
    def get_features(self, x):
        """Extract features before classifier."""
        B = x.shape[0]
        x = self.cnn_backbone(x)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.transformer(x)
        return self.norm(x[:, 0])


if __name__ == "__main__":
    # Test the model
    model = CNNViT(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"CNNViT Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
