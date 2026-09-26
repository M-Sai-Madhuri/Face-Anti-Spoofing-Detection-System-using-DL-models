"""
Ensemble Model - Bagging with Soft Voting
Combines predictions from all 4 models for robust anti-spoofing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

from .cdcn import CDCN
from .attention_cnn import AttentionCNN
from .frequency_cnn import FrequencyCNN
from .cnn_vit import CNNViT


class EnsembleModel(nn.Module):
    """
    Bagging Ensemble for Face Anti-Spoofing.
    
    Combines 4 diverse models:
    - CDCN: Texture-based detection
    - AttentionCNN: Efficient attention-based
    - FrequencyCNN: Frequency domain analysis
    - CNNViT: Global-local hybrid
    
    Uses soft voting with learnable/fixed weights.
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5,
                 weights: Optional[List[float]] = None,
                 models_to_use: Optional[List[str]] = None,
                 freq_version: str = 'v2'):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Default: use all models
        if models_to_use is None:
            models_to_use = ['cdcn', 'attention_cnn', 'frequency_cnn', 'cnn_vit']
        
        self.model_names = models_to_use
        self.models = nn.ModuleDict()
        
        # Initialize selected models
        if 'cdcn' in models_to_use:
            self.models['cdcn'] = CDCN(num_classes=num_classes, dropout=dropout)
        
        if 'attention_cnn' in models_to_use:
            self.models['attention_cnn'] = AttentionCNN(
                num_classes=num_classes, pretrained=pretrained, dropout=dropout
            )
        
        if 'frequency_cnn' in models_to_use:
            self.models['frequency_cnn'] = FrequencyCNN(
                num_classes=num_classes, dropout=dropout, version=freq_version
            )
        
        if 'cnn_vit' in models_to_use:
            self.models['cnn_vit'] = CNNViT(
                num_classes=num_classes, pretrained=pretrained, dropout=dropout/2
            )
        
        # Ensemble weights (learnable or fixed)
        num_models = len(self.models)
        if weights is None:
            weights = [1.0 / num_models] * num_models
        
        self.register_buffer('ensemble_weights', torch.tensor(weights))
        
        # Optionally learnable weights
        self.learnable_weights = nn.Parameter(torch.ones(num_models))
    
    def forward(self, x, use_learnable_weights=True):
        """
        Forward pass with ensemble voting.
        
        Args:
            x: Input tensor (B, 3, H, W)
            use_learnable_weights: If True, use learnable softmax weights
        
        Returns:
            Ensemble prediction (B, num_classes)
        """
        outputs = []
        
        for name in self.model_names:
            if name in self.models:
                out = self.models[name](x)
                outputs.append(out)
        
        # Stack outputs: (num_models, B, num_classes)
        outputs = torch.stack(outputs, dim=0)
        
        # Get weights
        if use_learnable_weights:
            weights = F.softmax(self.learnable_weights, dim=0)
        else:
            weights = self.ensemble_weights
        
        # Weighted average (soft voting)
        # weights: (num_models,) -> (num_models, 1, 1)
        weights = weights.view(-1, 1, 1)
        ensemble_output = (outputs * weights).sum(dim=0)
        
        return ensemble_output
    
    def forward_individual(self, x) -> Dict[str, torch.Tensor]:
        """Get individual model predictions."""
        outputs = {}
        for name in self.model_names:
            if name in self.models:
                outputs[name] = self.models[name](x)
        return outputs
    
    def load_individual_checkpoints(self, checkpoint_paths: Dict[str, str]):
        """Load pretrained weights for individual models."""
        for name, path in checkpoint_paths.items():
            if name in self.models:
                state_dict = torch.load(path, map_location='cpu')
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                self.models[name].load_state_dict(state_dict)
                print(f"Loaded checkpoint for {name} from {path}")
    
    def freeze_models(self, model_names: Optional[List[str]] = None):
        """Freeze specified models (or all if None)."""
        if model_names is None:
            model_names = list(self.models.keys())
        
        for name in model_names:
            if name in self.models:
                for param in self.models[name].parameters():
                    param.requires_grad = False
    
    def unfreeze_models(self, model_names: Optional[List[str]] = None):
        """Unfreeze specified models (or all if None)."""
        if model_names is None:
            model_names = list(self.models.keys())
        
        for name in model_names:
            if name in self.models:
                for param in self.models[name].parameters():
                    param.requires_grad = True
    
    def get_model_params(self) -> Dict[str, int]:
        """Get parameter count for each model."""
        params = {}
        for name, model in self.models.items():
            params[name] = sum(p.numel() for p in model.parameters())
        params['total'] = sum(params.values())
        return params


if __name__ == "__main__":
    # Test the ensemble
    model = EnsembleModel(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    
    # Test ensemble forward
    out = model(x)
    print(f"Ensemble Output shape: {out.shape}")
    
    # Test individual predictions
    individual = model.forward_individual(x)
    for name, pred in individual.items():
        print(f"  {name}: {pred.shape}")
    
    # Show parameter counts
    params = model.get_model_params()
    print("\nParameter counts:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")
