"""
Full visual comparison for 3 selected examples:
  airplane (result_00), apple (result_01), bench (result_05 from 30-run)

Style matches visual_comparison.py:
  - 96x96 cropped image
  - white background, blue overlay (0,0,255) alpha=0.5
  - columns: Input (center point) | GT | SAM3 | SAM2 | PicoSAM3 | PicoSAM2
  - dpi=300, bold 16pt titles
"""

import os, sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR   = os.path.join(ROOT_DIR, "checkpoints")
DATA_DIR   = "/datasets/pbonazzi/picosam3_data"
IMG_ROOT   = os.path.join(DATA_DIR, "val2017")
ANN_FILE   = os.path.join(DATA_DIR, "annotations", "instances_val2017.json")
OUT_DIR    = os.path.join(ROOT_DIR, "model_compression", "qualitative_results")
IMAGE_SIZE = 96
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, ROOT_DIR)
from model_compression.model import PicoSAM2
from model_compression.utils import pad_bbox_to_square

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── helpers ────────────────────────────────────────────────────────────────────

def resize_mask_to_roi(mask_96, roi_h, roi_w):
    t = torch.tensor(mask_96.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=(roi_h, roi_w), mode="nearest").squeeze().numpy() > 0.5

def overlay_on_full(full_img, mask_96, x1, y1, x2, y2, color=(0, 0, 255), alpha=0.5):
    """Overlay 96×96 mask onto the full image at ROI location with white border."""
    roi_h, roi_w = y2 - y1, x2 - x1
    mask_roi = resize_mask_to_roi(mask_96, roi_h, roi_w)
    out = full_img.copy().astype(float)
    region = out[y1:y2, x1:x2]
    region[mask_roi] = (1 - alpha) * region[mask_roi] + alpha * np.array(color)
    contour = mask_roi & ~binary_erosion(mask_roi)
    region[contour] = (255, 255, 255)
    out[y1:y2, x1:x2] = region
    return out.astype(np.uint8)

def add_zoom_inset(ax, img, x1, y1, x2, y2, loc='top_left'):
    """Add a zoom inset with connecting lines. loc: 'top_left' or 'top_right'."""
    x0 = 0.0 if loc == 'top_left' else 0.5
    axins = ax.inset_axes([x0, 0.5, 0.5, 0.5])
    axins.imshow(img)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y2, y1)
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor('gold')
        spine.set_linewidth(1.5)
    ax.indicate_inset_zoom(axins, edgecolor='gold', linewidth=1.5)

def plot_comparison(full_img_np, x1, y1, x2, y2, gt_mask_96, preds_dict, out_path,
                    show_titles=True, show_zoom=True, inset_loc='top_left'):
    cx_full = (x1 + x2) // 2
    cy_full = (y1 + y2) // 2

    all_panels = [("Ground Truth", gt_mask_96)] + list(preds_dict.items())
    ncols = 1 + len(all_panels)

    fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 5))

    # Input panel — full image, red dot at crop centre
    axs[0].imshow(full_img_np)
    axs[0].scatter([cx_full], [cy_full], c='red', s=40, zorder=5)
    if show_zoom:
        add_zoom_inset(axs[0], full_img_np, x1, y1, x2, y2, loc=inset_loc)
    if show_titles:
        axs[0].set_title("Input", fontsize=16, weight='bold', pad=20)
    axs[0].axis('off')

    # GT + model panels
    for ax, (label, mask_96) in zip(axs[1:], all_panels):
        overlaid = overlay_on_full(full_img_np, mask_96, x1, y1, x2, y2)
        ax.imshow(overlaid)
        if show_zoom:
            add_zoom_inset(ax, overlaid, x1, y1, x2, y2, loc=inset_loc)
        if show_titles:
            ax.set_title(label, fontsize=16, weight='bold', pad=20)
        ax.axis('off')

    plt.tight_layout()
    if show_titles:
        plt.subplots_adjust(top=0.85)
    plt.savefig(out_path, dpi=300, format="pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")

# ── load COCO ──────────────────────────────────────────────────────────────────
print("Loading COCO val2017 …")
coco = COCO(ANN_FILE)
cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

all_anns = [
    a for a in coco.loadAnns(coco.getAnnIds())
    if a.get("iscrowd", 0) == 0
    and "segmentation" in a
    and 2000 < a.get("area", 0) < 150_000
]

# reproduce 80-run selection (one per category, middle element, sorted alpha)
by_cat = defaultdict(list)
for a in all_anns:
    by_cat[a["category_id"]].append(a)
per_cat_sorted = [
    anns[len(anns) // 2]
    for anns in sorted(by_cat.values(), key=lambda v: cat_id_to_name[v[0]["category_id"]])
]
ann_airplane = per_cat_sorted[0]
ann_apple    = per_cat_sorted[1]
ann_zebra    = per_cat_sorted[79]

# reproduce 30-run selection for bench (evenly-spaced, index 5)
step_30   = max(1, len(all_anns) // 30)
ann_bench = all_anns[5 * step_30]

TARGETS = [
    # (cat_name, ann, show_titles, show_zoom, inset_loc)
    ("airplane", ann_airplane, True,  False, 'top_left'),
    ("apple",    ann_apple,    False, True,  'top_left'),
    ("bench",    ann_bench,    False, True,  'top_left'),
    ("zebra",    ann_zebra,    False, False, 'top_left'),
]

# ── load PicoSAM3 & PicoSAM2 ──────────────────────────────────────────────────
print("Loading PicoSAM3 (supervised) …")
picosam3 = PicoSAM2().to(DEVICE)
picosam3.load_state_dict(torch.load(os.path.join(CKPT_DIR, "PicoSAM3_epoch1.pt"), map_location=DEVICE))
picosam3.eval()

print("Loading PicoSAM2 (distilled) …")
picosam2 = PicoSAM2().to(DEVICE)
picosam2.load_state_dict(torch.load(os.path.join(CKPT_DIR, "PicoSAM3_student_epoch1.pt"), map_location=DEVICE))
picosam2.eval()

# ── load SAM2 ─────────────────────────────────────────────────────────────────
print("Loading SAM2.1 Large …")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sam2_predictor = SAM2ImagePredictor(
    build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",
               os.path.join(CKPT_DIR, "sam2.1_hiera_large.pt"),
               device=DEVICE, mode="eval")
)

# ── load SAM3 ─────────────────────────────────────────────────────────────────
print("Loading SAM3 …")
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
sam3_model = build_sam3_image_model(
    load_from_HF=True, device=str(DEVICE), eval_mode=True, enable_segmentation=True
)
sam3_proc = Sam3Processor(model=sam3_model, device=DEVICE, confidence_threshold=0.0)

print("All models loaded.\n")

# ── run and save ───────────────────────────────────────────────────────────────
for cat_name, ann, show_titles, show_zoom, inset_loc in TARGETS:
    print(f"Processing {cat_name} (img {ann['image_id']}) …")

    # load image and compute ROI crop
    img_info  = coco.loadImgs(ann["image_id"])[0]
    img_path  = os.path.join(IMG_ROOT, img_info["file_name"])
    image_pil = Image.open(img_path).convert("RGB")
    img_np    = np.array(image_pil)           # uint8 full image
    img_h, img_w = img_np.shape[:2]

    x1, y1, x2, y2 = pad_bbox_to_square(ann["bbox"], img_w, img_h, padding=0.1)
    crop_pil  = image_pil.crop((x1, y1, x2, y2))
    crop_96   = crop_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    crop_np   = np.array(crop_96)             # uint8, HxWx3
    crop_t    = normalize(crop_96).unsqueeze(0).to(DEVICE)

    # GT mask resized to 96x96
    gt_full   = coco.annToMask(ann).astype(np.float32)
    gt_crop   = gt_full[y1:y2, x1:x2]
    gt_t      = torch.tensor(gt_crop).unsqueeze(0).unsqueeze(0)
    gt_96     = F.interpolate(gt_t, size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest").squeeze().numpy() > 0.5

    # center point on 96x96 crop
    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2
    pt  = np.array([[[cx, cy]]])
    lbl = np.array([[1]])

    # ── PicoSAM3 & PicoSAM2 ──
    with torch.no_grad():
        ps3_mask = (torch.sigmoid(picosam3(crop_t)).squeeze().cpu().numpy() > 0.5)
        ps2_mask = (torch.sigmoid(picosam2(crop_t)).squeeze().cpu().numpy() > 0.5)

    # ── SAM2 (center point on 96x96 crop) ──
    sam2_predictor.set_image(crop_np)
    with torch.inference_mode(), torch.autocast(str(DEVICE), dtype=torch.bfloat16):
        masks, scores, _ = sam2_predictor.predict(
            point_coords=pt, point_labels=lbl, multimask_output=True
        )
    sam2_mask = masks[np.argmax(scores)].astype(bool)

    # ── SAM3 (text prompt on 96x96 crop) ──
    state = sam3_proc.set_image(crop_96)
    state = sam3_proc.set_text_prompt(prompt=cat_name, state=state)
    if "masks" in state and len(state["masks"]) > 0:
        best = int(state["scores"].argmax())
        sam3_mask = state["masks"][best, 0].cpu().numpy().astype(bool)
    else:
        sam3_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)

    preds = {
        "SAM2.1 L": sam2_mask,
        "SAM3":     sam3_mask,
        "PicoSAM2": ps2_mask,
        "PicoSAM3": ps3_mask,
    }

    out_path = os.path.join(OUT_DIR, f"comparison_{cat_name}.pdf")
    plot_comparison(img_np, x1, y1, x2, y2, gt_96, preds, out_path,
                    show_titles=show_titles, show_zoom=show_zoom, inset_loc=inset_loc)

print("\nDone.")
