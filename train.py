# -*- coding: utf-8 -*-

# train.py
# Main script to train the U-Net model.

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt

# Custom imports
from models.unet import UNet
from utils.dataset import NanoparticleDataset
from utils.transforms import get_train_transforms, get_val_transforms

# Function to calculate Dice score for validation
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    accuracy = num_correct / num_pixels * 100
    dice = dice_score / len(loader)
    print(f"Validation Accuracy: {accuracy:.2f}%")
    print(f"Dice Score: {dice:.4f}")
    model.train()
    return dice

# The main training loop for one epoch
def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(loader, leave=True)
    mean_loss = []

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # Forward pass
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        mean_loss.append(loss.item())
        loop.set_postfix(loss=loss.item())

    return sum(mean_loss)/len(mean_loss)


def main():
    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(description='Train U-Net for Nanoparticle Segmentation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--height', type=int, default=256, help='Image height')
    parser.add_argument('--width', type=int, default=256, help='Image width')
    parser.add_argument('--load-model', action='store_true', help='Load a pre-trained model')
    args = parser.parse_args()

    # --- Setup ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    TRAIN_IMG_DIR = "data/train/images/"
    TRAIN_MASK_DIR = "data/train/masks/"
    VAL_IMG_DIR = "data/val/images/"
    VAL_MASK_DIR = "data/val/masks/"
    MODEL_SAVE_PATH = "saved_models/best_model.pth"

    os.makedirs("saved_models", exist_ok=True)

    # --- DataLoaders ---
    train_dataset = NanoparticleDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        transform=get_train_transforms(args.height, args.width),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
    )

    val_dataset = NanoparticleDataset(
        image_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        transform=get_val_transforms(args.height, args.width),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
    )

    # --- Model, Loss, Optimizer ---
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    if args.load_model and os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print("Loaded pre-trained model.")

    loss_fn = torch.nn.BCEWithLogitsLoss() # Good for binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    # --- Training Loop ---
    best_dice_score = -1
    train_losses = []

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, DEVICE)
        train_losses.append(avg_loss)

        # Check accuracy
        dice_score = check_accuracy(val_loader, model, device=DEVICE)

        # Save model if it's the best one so far
        if dice_score > best_dice_score:
            best_dice_score = dice_score
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("=> New best model saved!")

    # --- Plot and save the training loss curve ---
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('saved_models/loss_curve.png')
    print("Loss curve saved to saved_models/loss_curve.png")

if __name__ == "__main__":
    main()