# src/models/__init__.py
"""Model architectures for Face Anti-Spoofing"""

from .cdcn import CDCN
from .attention_cnn import AttentionCNN
from .frequency_cnn import FrequencyCNN
from .cnn_vit import CNNViT
from .ensemble import EnsembleModel

__all__ = ['CDCN', 'AttentionCNN', 'FrequencyCNN', 'CNNViT', 'EnsembleModel']


def get_model(name: str, **kwargs):
    """Factory function to get model by name."""
    models = {
        'cdcn': CDCN,
        'attention_cnn': AttentionCNN,
        'frequency_cnn': FrequencyCNN,
        'cnn_vit': CNNViT,
        'ensemble': EnsembleModel,
    }
    
    if name == 'ensemble':
        return models[name](num_classes=kwargs.get('num_classes', 2),
                           pretrained=kwargs.get('pretrained', True),
                           dropout=kwargs.get('dropout', 0.5),
                           freq_version=kwargs.get('version', kwargs.get('freq_version', 'v2')))
    
    # Filter kwargs for models that don't support 'version'
    if name != 'frequency_cnn' and 'version' in kwargs:
        kwargs.pop('version')
        
    return models[name](**kwargs)
