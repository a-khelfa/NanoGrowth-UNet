# -*- coding: utf-8 -*-

# generate_synthetic_data.py
# This script generates a synthetic dataset for training and validation.
# It creates simple shapes to simulate nanoparticles and adds noise to them.

import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def create_synthetic_image(height, width, num_particles):
    """
    Creates a single synthetic image with corresponding mask.

    Returns:
        tuple: A tuple containing the noisy image and the clean mask.
    """
    # Create a black background for both image and mask
    mask = np.zeros((height, width), dtype=np.uint8)
    image = np.zeros((height, width), dtype=np.float32)

    for _ in range(num_particles):
        # Randomly choose position
        center_x = np.random.randint(20, width - 20)
        center_y = np.random.randint(20, height - 20)

        # Randomly choose size (ellipse axes)
        axis1 = np.random.randint(5, 20)
        axis2 = np.random.randint(5, 20)

        # Randomly choose orientation and intensity
        angle = np.random.randint(0, 180)
        color = np.random.randint(150, 255)

        # Draw the particle on the mask (white on black)
        cv2.ellipse(mask, (center_x, center_y), (axis1, axis2), angle, 0, 360, 255, -1)
        # Draw the particle on the image with its intensity
        cv2.ellipse(image, (center_x, center_y), (axis1, axis2), angle, 0, 360, color, -1)

    # --- Simulate Microscope Effects ---
    # Apply Gaussian blur to simulate optical effects
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Add Gaussian noise
    noise = np.random.normal(0, 25, image.shape).astype(np.float32)
    noisy_image = image + noise

    # Clip values to be in the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image, mask

def generate_dataset(num_images, path, particle_range=(3, 15)):
    """
    Generates and saves a dataset of synthetic images.
    """
    img_path = os.path.join(path, "images")
    mask_path = os.path.join(path, "masks")
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)

    print(f"Generating {num_images} images in {path}...")
    for i in tqdm(range(num_images)):
        num_particles = np.random.randint(particle_range[0], particle_range[1])
        image, mask = create_synthetic_image(256, 256, num_particles)

        filename = f"synthetic_{i:04d}.png"
        cv2.imwrite(os.path.join(img_path, filename), image)
        cv2.imwrite(os.path.join(mask_path, filename), mask)

def main():
    parser = argparse.ArgumentParser(description='Generate a synthetic dataset for U-Net training.')
    parser.add_argument('--train-count', type=int, default=200, help='Number of images for the training set.')
    parser.add_argument('--val-count', type=int, default=40, help='Number of images for the validation set.')
    args = parser.parse_args()

    # Define paths
    TRAIN_PATH = "data/train/"
    VAL_PATH = "data/val/"

    # Generate datasets
    generate_dataset(args.train_count, TRAIN_PATH)
    generate_dataset(args.val_count, VAL_PATH)

    print("\nSynthetic dataset generated successfully!")
    print(f"Training data in: {TRAIN_PATH}")
    print(f"Validation data in: {VAL_PATH}")

if __name__ == "__main__":
    main()