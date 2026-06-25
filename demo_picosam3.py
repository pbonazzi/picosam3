"""
PicoSAM3 demo — runs on a single image with a bounding-box prompt.
No COCO dataset required.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from PIL import Image
from scipy.ndimage import binary_erosion

from model_compression.model import PicoSAM2 as PicoSAM3
from model_compression.utils import pad_bbox_to_square

CKPT_PATH  = os.path.join(os.path.dirname(__file__), "checkpoints", "PicoSAM3_student_epoch1.pt")
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "demo", "data", "sample_dog.jpg")
OUT_PATH   = os.path.join(os.path.dirname(__file__), "demo", "data", "demo_result.png")
IMAGE_SIZE = 96
DEVICE     = torch.device("cpu")

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def overlay_mask(img_np, mask_np, color=(0.2, 0.6, 1.0), alpha=0.45):
    out = img_np.copy()
    for c, v in enumerate(color):
        out[..., c] = np.where(mask_np, (1 - alpha) * out[..., c] + alpha * v, out[..., c])
    contour = mask_np & ~binary_erosion(mask_np)
    out[contour] = (1.0, 1.0, 1.0)
    return out

print(f"Loading PicoSAM3 from {CKPT_PATH} ...")
model = PicoSAM3().to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print("Model loaded.")

image_pil = Image.open(IMAGE_PATH).convert("RGB")
image_np  = np.array(image_pil) / 255.0
img_h, img_w = image_np.shape[:2]

# Center bounding box covering ~60 % of the image (the dog is roughly centered)
cx, cy = img_w // 2, img_h // 2
bw, bh = int(img_w * 0.60), int(img_h * 0.60)
bx, by = cx - bw // 2, cy - bh // 2
bbox   = [bx, by, bw, bh]   # x, y, w, h  (COCO format)

x1, y1, x2, y2 = pad_bbox_to_square(bbox, img_w, img_h, padding=0.1)
roi_w, roi_h   = x2 - x1, y2 - y1

crop    = image_pil.crop((x1, y1, x2, y2))
crop_rs = crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
inp     = normalize(crop_rs).unsqueeze(0).to(DEVICE)

print("Running inference ...")
with torch.no_grad():
    logits   = model(inp)
    prob     = torch.sigmoid(logits)
    prob_roi = F.interpolate(prob, size=(roi_h, roi_w), mode="bilinear", align_corners=False)
    mask_roi = (prob_roi.squeeze().cpu().numpy() > 0.5)

mask_full = np.zeros((img_h, img_w), dtype=bool)
mask_full[y1:y2, x1:x2] = mask_roi

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#1a1a2e")

axes[0].imshow(image_np)
rect = patches.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor="#f5a623", facecolor="none")
axes[0].add_patch(rect)
axes[0].set_title("Input + BBox Prompt", color="white", fontsize=13, pad=8)
axes[0].axis("off")

axes[1].imshow(overlay_mask(image_np, mask_full))
axes[1].set_title("PicoSAM3 Prediction", color="white", fontsize=13, pad=8)
axes[1].axis("off")

fig.suptitle("PicoSAM3 Student Demo — dog", color="white", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Result saved → {OUT_PATH}")
