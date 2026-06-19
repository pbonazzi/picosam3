"""
Qualitative results for PicoSAM3: runs 10 COCO val2017 examples and saves
side-by-side visualizations of the original image (with bbox prompt) and the
predicted mask overlaid on the full image.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.join(BASE_DIR, "..")
CKPT_PATH   = os.path.join(ROOT_DIR, "checkpoints", "PicoSAM3_epoch1.pt")
DATA_DIR    = "/datasets/pbonazzi/picosam3_data"
IMG_ROOT    = os.path.join(DATA_DIR, "val2017")
ANN_FILE    = os.path.join(DATA_DIR, "annotations", "instances_val2017.json")
OUT_DIR     = os.path.join(BASE_DIR, "qualitative_results")
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_SIZE  = 96
N_EXAMPLES  = 80
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, ROOT_DIR)
from model_compression.model import PicoSAM2 as PicoSAM3
from model_compression.utils import pad_bbox_to_square

# ── transforms ─────────────────────────────────────────────────────────────────
normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denorm(t):
    return (t.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).permute(1, 2, 0).numpy()

def overlay_mask(img_np, mask_np, color=(0.2, 0.6, 1.0), alpha=0.45):
    """Blend a binary mask onto an RGB image (values in [0,1])."""
    out = img_np.copy()
    for c, v in enumerate(color):
        out[..., c] = np.where(mask_np, (1 - alpha) * out[..., c] + alpha * v, out[..., c])
    # Draw a contour
    from scipy.ndimage import binary_erosion
    contour = mask_np & ~binary_erosion(mask_np)
    out[contour] = (1.0, 1.0, 1.0)
    return out

# ── load model ─────────────────────────────────────────────────────────────────
print(f"Loading PicoSAM3 from {CKPT_PATH} …")
model = PicoSAM3().to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()
print("Model loaded.")

# ── load COCO val2017 ──────────────────────────────────────────────────────────
print("Loading COCO val2017 annotations …")
coco = COCO(ANN_FILE)
cat_id_to_name = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}

# Filter for non-crowd annotations with reasonably large objects
all_anns = [
    ann for ann in coco.loadAnns(coco.getAnnIds())
    if ann.get("iscrowd", 0) == 0
    and "segmentation" in ann
    and ann.get("area", 0) > 2000          # skip tiny objects
    and ann.get("area", 0) < 150_000       # skip full-scene annotations
]

# Pick one annotation per category (sorted by cat name), then pad with evenly-spaced ones
from collections import defaultdict
by_cat = defaultdict(list)
for ann in all_anns:
    by_cat[ann["category_id"]].append(ann)

# One representative per category (pick the middle one for variety), sorted by name
samples = [
    anns[len(anns) // 2]
    for anns in sorted(by_cat.values(), key=lambda a: cat_id_to_name[a[0]["category_id"]])
][:N_EXAMPLES]
print(f"Selected {len(samples)} examples.")

# ── run inference and save ─────────────────────────────────────────────────────

with torch.no_grad():
    for idx, ann in enumerate(samples):
        img_info = coco.loadImgs(ann["image_id"])[0]
        img_path = os.path.join(IMG_ROOT, img_info["file_name"])

        image_pil  = Image.open(img_path).convert("RGB")
        image_np   = np.array(image_pil) / 255.0      # H×W×3, float [0,1]
        img_h, img_w = image_np.shape[:2]

        # Compute square ROI
        x1, y1, x2, y2 = pad_bbox_to_square(ann["bbox"], img_w, img_h, padding=0.1)
        roi_w, roi_h    = x2 - x1, y2 - y1

        # Crop and resize to 96×96
        crop      = image_pil.crop((x1, y1, x2, y2))
        crop_rs   = crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        inp       = normalize(crop_rs).unsqueeze(0).to(DEVICE)   # 1×3×96×96

        # PicoSAM3 inference
        logits    = model(inp)                                    # 1×1×96×96
        prob      = torch.sigmoid(logits)                         # same shape

        # Resize predicted mask back to the ROI footprint on the original image
        prob_roi  = F.interpolate(prob, size=(roi_h, roi_w), mode="bilinear", align_corners=False)
        mask_roi  = (prob_roi.squeeze().cpu().numpy() > 0.5)

        # Place mask on the full-image canvas
        mask_full = np.zeros((img_h, img_w), dtype=bool)
        mask_full[y1:y2, x1:x2] = mask_roi

        # ── figure: original | masked ──────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#1a1a2e")

        # Panel 1 – original image with bounding-box prompt
        axes[0].imshow(image_np)
        bx, by, bw, bh = ann["bbox"]
        rect = patches.Rectangle(
            (bx, by), bw, bh,
            linewidth=2, edgecolor="#f5a623", facecolor="none",
        )
        axes[0].add_patch(rect)
        axes[0].set_title("Input + BBox Prompt", color="white", fontsize=13, pad=8)
        axes[0].axis("off")

        # Panel 2 – original image with predicted mask overlay
        masked_np = overlay_mask(image_np, mask_full)
        axes[1].imshow(masked_np)
        axes[1].set_title("PicoSAM3 Prediction", color="white", fontsize=13, pad=8)
        axes[1].axis("off")

        # Category label
        cat_name = cat_id_to_name.get(ann.get("category_id", 0), "object")
        fig.suptitle(
            f'"{cat_name}"  ·  COCO img {ann["image_id"]}',
            color="white", fontsize=14, y=1.01,
        )

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"result_{idx:02d}_{cat_name.replace(' ', '_')}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  [{idx+1}/{N_EXAMPLES}] Saved → {out_path}")

print(f"\nAll results saved to {OUT_DIR}/")
