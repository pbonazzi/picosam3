import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from pycocotools.coco import COCO
from picosam2_model_distillation import PicoSAM2
from picosam2_model_distillation import PicoSAM2Dataset
from model_compression.utils import bce_dice_loss, compute_iou


IMAGE_SIZE = 96
BATCH_SIZE = 8
NUM_EPOCHS = 1
LEARNING_RATE = 3e-4
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_ROOT = os.path.join(BASE_DIR, "../dataset/train2017")
ANN_FILE = os.path.join(BASE_DIR, "../dataset/annotations/instances_train2017.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "../checkpoints")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def train():
    wandb.init(project="PicoSAM2-scratch", config={"img_size": IMAGE_SIZE, "epochs": NUM_EPOCHS})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PicoSAM2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / 1000))

    dataset = PicoSAM2Dataset(IMG_ROOT, ANN_FILE, IMAGE_SIZE)
    train_len = int(len(dataset) * 0.95)
    train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, total_iou, samples = 0, 0, 0

        for batch_idx, (images, masks, _, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} - Train")):
            images, masks = images.to(device), masks.to(device)

            preds = model(images)
            loss = bce_dice_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_iou = compute_iou(preds, masks)
            wandb.log({"batch_loss": loss.item(), "batch_mIoU": batch_iou, "epoch": epoch + 1})

            total_loss += loss.item() * images.size(0)
            total_iou += batch_iou * images.size(0)
            samples += images.size(0)

        wandb.log({"train_loss": total_loss / samples, "train_mIoU": total_iou / samples, "epoch": epoch + 1})

        model.eval()
        val_loss, val_iou, val_samples = 0, 0, 0
        with torch.no_grad():
            for images, masks, _, _ in tqdm(val_loader, desc=f"Epoch {epoch + 1} - Val"):
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                loss = bce_dice_loss(preds, masks)
                val_loss += loss.item() * images.size(0)
                val_iou += compute_iou(preds, masks) * images.size(0)
                val_samples += images.size(0)

        wandb.log({"val_loss": val_loss / val_samples, "val_mIoU": val_iou / val_samples, "epoch": epoch + 1})

        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"PicoSAM2_epoch{epoch + 1}.pt"))


if __name__ == "__main__":
    train()
