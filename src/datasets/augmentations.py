"""
Data Augmentations for Face Anti-Spoofing
Includes both training and validation transforms.
"""

import torch
from torchvision import transforms


def get_train_transforms(
    image_size: int = 224,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Get training data augmentations.
    
    Includes:
    - Random horizontal flip
    - Random rotation
    - Color jitter
    - Random resized crop
    - Gaussian blur (occasional)
    - Normalization
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transforms(
    image_size: int = 224,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Get validation/test data transforms.
    
    Only includes:
    - Resize
    - Center crop
    - Normalization
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_inference_transforms(
    image_size: int = 224,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Get inference transforms (same as validation).
    """
    return get_val_transforms(image_size, mean, std)


def denormalize(
    tensor: torch.Tensor,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Denormalize a tensor for visualization.
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    if tensor.device != mean.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    
    return tensor * std + mean


if __name__ == "__main__":
    # Test transforms
    from PIL import Image
    import numpy as np
    
    # Create dummy image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    
    train_tf = get_train_transforms()
    val_tf = get_val_transforms()
    
    train_out = train_tf(dummy_img)
    val_out = val_tf(dummy_img)
    
    print(f"Train transform output shape: {train_out.shape}")
    print(f"Val transform output shape: {val_out.shape}")
