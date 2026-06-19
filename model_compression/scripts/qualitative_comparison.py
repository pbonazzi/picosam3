import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import random
from scipy import ndimage

# Configuration
BASE_DIR = "/datasets/pbonazzi/picosam3_data/"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
IMG_ROOT = os.path.join(BASE_DIR, "val2017")
COCO_ANN_FILE = os.path.join(BASE_DIR, "annotations/instances_val2017.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "images", "qualitative_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pad_bbox_to_square(bbox, img_width, img_height, padding=0.1):
    """Pad bbox and make it square."""
    x, y, w, h = bbox
    pad_w, pad_h = w * padding, h * padding
    x, y = x - pad_w, y - pad_h
    w, h = w + 2 * pad_w, h + 2 * pad_h

    size = max(w, h)
    cx, cy = x + w / 2, y + h / 2
    x1, y1 = cx - size / 2, cy - size / 2
    x2, y2 = x1 + size, y1 + size

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_width, x2), min(img_height, y2)

    return int(x1), int(y1), int(x2), int(y2)


def get_mask_border(mask, thickness=2):
    """Get border of binary mask."""
    mask_bool = mask > 0.5
    dilated = ndimage.binary_dilation(mask_bool, iterations=thickness)
    eroded = ndimage.binary_erosion(mask_bool, iterations=1)
    return dilated & ~eroded


def apply_mask_overlay(image, mask, fill_color=(70, 130, 255), border_color=(255, 255, 255), alpha=0.5, border_thickness=2):
    """Apply mask overlay with border."""
    img = image.copy().astype(np.float32)
    mask_bool = mask > 0.5

    for c in range(3):
        img[:, :, c] = np.where(mask_bool, img[:, :, c] * (1 - alpha) + fill_color[c] * alpha, img[:, :, c])

    if border_thickness > 0:
        border = get_mask_border(mask, thickness=border_thickness)
        for c in range(3):
            img[:, :, c] = np.where(border, border_color[c], img[:, :, c])

    return img.astype(np.uint8)


def normalize_image(pil_img):
    """Normalize PIL image for model input."""
    img_tensor = torch.tensor(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
    img_tensor = (img_tensor - MEAN) / STD
    return img_tensor.unsqueeze(0)


def run_picosam(model, image_crop_pil, device):
    """Run PicoSAM inference."""
    img_resized = image_crop_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    img_tensor = normalize_image(img_resized).to(device)

    with torch.no_grad():
        pred = model(img_tensor)

    pred_mask = torch.sigmoid(pred[0, 0]).cpu().numpy()
    return pred_mask


def run_sam2(predictor, image_np, bbox):
    """Run SAM2 inference with box prompt."""
    predictor.set_image(image_np)

    x, y, w, h = bbox
    box = np.array([x, y, x + w, y + h])

    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=True
    )

    best_idx = np.argmax(scores)
    return masks[best_idx]


def run_sam3(processor, pil_image, bbox, img_w, img_h):
    """Run SAM3 inference with box prompt."""
    # Convert bbox to normalized cxcywh
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    box_normalized = [cx, cy, nw, nh]

    state = processor.set_image(pil_image)
    processor.reset_all_prompts(state)
    state = processor.add_geometric_prompt(box=box_normalized, label=True, state=state)

    if "masks" not in state or len(state["masks"]) == 0:
        return np.zeros((img_h, img_w), dtype=np.float32)

    scores = state["scores"].cpu().numpy()
    best_idx = scores.argmax()
    return state["masks"][best_idx, 0].cpu().float().numpy()


def create_qualitative_comparison(samples, models, output_name="qualitative_comparison"):
    """Create horizontal banner comparison figure."""

    num_samples = len(samples)
    num_cols = 2 + len(models)  # Original + GT + models

    fig, axes = plt.subplots(num_samples, num_cols, figsize=(3 * num_cols, 3 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Input", "Ground Truth"] + list(models.keys())

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=14, fontweight='bold', pad=10)

    for row, sample in enumerate(samples):
        image_crop = sample["image_crop"]
        gt_mask = sample["gt_mask"]
        image_np = np.array(image_crop)

        # Column 0: Original cropped image
        axes[row, 0].imshow(image_crop)
        axes[row, 0].axis('off')

        # Column 1: Ground truth
        gt_overlay = apply_mask_overlay(image_np, gt_mask, fill_color=(0, 200, 0), alpha=0.5)
        axes[row, 1].imshow(gt_overlay)
        axes[row, 1].axis('off')

        # Model predictions
        for col, (model_name, pred_mask) in enumerate(sample["predictions"].items(), start=2):
            pred_overlay = apply_mask_overlay(image_np, pred_mask, fill_color=(70, 130, 255), alpha=0.5)
            axes[row, col].imshow(pred_overlay)
            axes[row, col].axis('off')

    plt.tight_layout()

    # Save
    png_path = os.path.join(OUTPUT_DIR, f"{output_name}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{output_name}.pdf")
    plt.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    print("Loading models...")

    # Import models
    from model_compression.model import PicoSAM2, PicoSAM3
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # Load PicoSAM models
    picosam2 = PicoSAM2().to(DEVICE)
    picosam2.load_state_dict(torch.load(os.path.join(CKPT_DIR, "PicoSAM2_SAM3_student_epoch1.pt"), map_location=DEVICE))
    picosam2.eval()

    picosam3 = PicoSAM3().to(DEVICE)
    picosam3.load_state_dict(torch.load(os.path.join(CKPT_DIR, "PicoSAM3_SAM3_student_epoch1.pt"), map_location=DEVICE))
    picosam3.eval()

    # Load SAM2
    sam2_model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        os.path.join(CKPT_DIR, "sam2.1_hiera_large.pt"),
        device=DEVICE,
        mode="eval"
    )
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # Load SAM3
    sam3_model = build_sam3_image_model(
        device="cuda",
        eval_mode=True,
        load_from_HF=True,
        enable_segmentation=True,
    )
    sam3_processor = Sam3Processor(model=sam3_model, device="cuda", confidence_threshold=0.0)

    print("Loading COCO annotations...")
    coco = COCO(COCO_ANN_FILE)

    # Get valid annotations (skip humans)
    person_cat_ids = coco.getCatIds(catNms=["person"])
    all_anns = [
        ann for ann in coco.loadAnns(coco.getAnnIds())
        if "segmentation" in ann and ann.get("iscrowd", 0) == 0 and ann["area"] > 10000
        and ann["category_id"] not in person_cat_ids
    ]

    # Filter to existing images
    existing_anns = []
    for ann in all_anns:
        img_info = coco.loadImgs(ann["image_id"])[0]
        img_path = os.path.join(IMG_ROOT, img_info["file_name"])
        if os.path.exists(img_path):
            existing_anns.append(ann)

    print(f"Found {len(existing_anns)} valid annotations")

    # Sample 3 images
    random.seed(1234)
    #selected_indices = random.sample(range(len(existing_anns)), min(3, len(existing_anns)))
    #selected_indices = [15, 785, 2374]
    selected_indices = [957, 785, 2374]
    selected_anns = [existing_anns[i] for i in selected_indices]
    print(f"Selected annotation indices: {selected_indices}")

    samples = []

    for i, ann in enumerate(selected_anns):
        print(f"\nProcessing sample {i+1}/{len(selected_anns)}...")

        img_info = coco.loadImgs(ann["image_id"])[0]
        img_path = os.path.join(IMG_ROOT, img_info["file_name"])

        pil_image = Image.open(img_path).convert("RGB")
        image_np = np.array(pil_image)
        img_h, img_w = image_np.shape[:2]

        bbox = ann["bbox"]
        gt_mask_full = coco.annToMask(ann)

        # Get crop region
        x1, y1, x2, y2 = pad_bbox_to_square(bbox, img_w, img_h, padding=0.1)

        # Crop image and mask
        image_crop = pil_image.crop((x1, y1, x2, y2))
        gt_mask_crop = gt_mask_full[y1:y2, x1:x2]

        crop_h, crop_w = gt_mask_crop.shape

        # Resize for display
        display_size = 256
        image_crop_display = image_crop.resize((display_size, display_size), Image.BILINEAR)
        gt_mask_display = np.array(Image.fromarray((gt_mask_crop * 255).astype(np.uint8)).resize(
            (display_size, display_size), Image.NEAREST)) / 255.0

        predictions = {}

        # SAM2
        print("  Running SAM2...")
        pred_sam2_full = run_sam2(sam2_predictor, image_np, bbox)
        pred_sam2_crop = pred_sam2_full[y1:y2, x1:x2].astype(np.float32)
        pred_sam2_display = np.array(Image.fromarray((pred_sam2_crop * 255).astype(np.uint8)).resize(
            (display_size, display_size), Image.NEAREST)) / 255.0
        predictions["SAM2.1 L"] = pred_sam2_display

        # SAM3
        print("  Running SAM3...")
        pred_sam3_full = run_sam3(sam3_processor, pil_image, bbox, img_w, img_h)
        pred_sam3_crop = pred_sam3_full[y1:y2, x1:x2]
        pred_sam3_display = np.array(Image.fromarray((pred_sam3_crop * 255).astype(np.uint8)).resize(
            (display_size, display_size), Image.NEAREST)) / 255.0
        predictions["SAM3"] = pred_sam3_display

        # PicoSAM2
        print("  Running PicoSAM2...")
        pred_picosam2 = run_picosam(picosam2, image_crop, DEVICE)
        pred_picosam2_display = np.array(Image.fromarray((pred_picosam2 * 255).astype(np.uint8)).resize(
            (display_size, display_size), Image.BILINEAR)) / 255.0
        predictions["PicoSAM2"] = pred_picosam2_display

        # PicoSAM3
        print("  Running PicoSAM3...")
        pred_picosam3 = run_picosam(picosam3, image_crop, DEVICE)
        pred_picosam3_display = np.array(Image.fromarray((pred_picosam3 * 255).astype(np.uint8)).resize(
            (display_size, display_size), Image.BILINEAR)) / 255.0
        predictions["PicoSAM3"] = pred_picosam3_display

        samples.append({
            "image_crop": image_crop_display,
            "gt_mask": gt_mask_display,
            "predictions": predictions,
        })

    print("\nCreating comparison figure...")
    create_qualitative_comparison(samples, {"SAM2.1 L": None, "SAM3": None, "PicoSAM2": None, "PicoSAM3": None})

    print("\nDone!")


if __name__ == "__main__":
    main()