# -*- coding: utf-8 -*-

# utils/transforms.py
# This file defines the data augmentation pipelines.

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(height, width):
    """
    Returns the transformation pipeline for the training dataset.
    Includes resizing, augmentation, and normalization.
    """
    train_transform = A.Compose(
        [
            A.Resize(height=height, width=width, always_apply=True),
            A.Rotate(limit=35, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
            # Normalize to [0, 1] then convert to tensor
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    return train_transform

def get_val_transforms(height, width):
    """
    Returns the transformation pipeline for the validation dataset.
    Includes resizing and normalization, but no augmentation.
    """
    val_transforms = A.Compose(
        [
            A.Resize(height=height, width=width, always_apply=True),
            # Normalize to [0, 1] then convert to tensor
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    return val_transforms