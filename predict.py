# -*- coding: utf-8 -*-

# predict.py
# Script to run inference on new images using a trained model.

import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import argparse
import os
from tqdm import tqdm

# Custom imports
from models.unet import UNet

def predict_single_image(model, image_path, device, height, width):
    """Runs prediction on a single image."""
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert("L")
    original_size = image.size

    # Apply transformations similar to validation
    img_tensor = TF.to_tensor(image)
    img_tensor = TF.resize(img_tensor, [height, width])
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        # Get prediction
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

        # Convert to numpy and resize to original image size
        pred_np = pred.squeeze(0).cpu().numpy()
        pred_img = Image.fromarray((pred_np[0] * 255).astype(np.uint8))
        pred_img = pred_img.resize(original_size, Image.NEAREST)

    return pred_img

def main():
    parser = argparse.ArgumentParser(description='Predict masks for input images')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--input', type=str, required=True, help='Path to an input image or a folder of images')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output mask(s)')
    parser.add_argument('--height', type=int, default=256, help='Image height for the model')
    parser.add_argument('--width', type=int, default=256, help='Image width for the model')
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Check if input is a directory or a single file
    if os.path.isdir(args.input):
        image_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        print(f"Found {len(image_files)} images to process.")

        for img_name in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(args.input, img_name)
            output_mask = predict_single_image(model, img_path, DEVICE, args.height, args.width)

            # Save the output mask
            output_path = os.path.join(args.output, f"mask_{img_name}")
            output_mask.save(output_path)

    elif os.path.isfile(args.input):
        print("Processing a single image.")
        output_mask = predict_single_image(model, args.input, DEVICE, args.height, args.width)

        # Save the output mask
        base_name = os.path.basename(args.input)
        output_path = os.path.join(args.output, f"mask_{base_name}")
        output_mask.save(output_path)
        print(f"Mask saved to {output_path}")

    else:
        print(f"Error: Input path '{args.input}' is not a valid file or directory.")

if __name__ == "__main__":
    main()