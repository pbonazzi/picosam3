import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import random

# Configuration
BASE_DIR = "/datasets/pbonazzi/picosam3_data/"
IMG_ROOT = os.path.join(BASE_DIR, "val2017")  # Both COCO and LVIS use same images
COCO_ANN_FILE = os.path.join(BASE_DIR, "annotations/instances_val2017.json")
LVIS_ANN_FILE = os.path.join(BASE_DIR, "annotations/lvis_v1_val.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "images", "coco_visualization")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def pad_bbox_to_square(bbox, img_width, img_height, padding=0.1):
    """Pad bbox by percentage and make it square, centered on original bbox."""
    x, y, w, h = bbox

    # Add padding
    pad_w = w * padding
    pad_h = h * padding
    x -= pad_w
    y -= pad_h
    w += 2 * pad_w
    h += 2 * pad_h

    # Make square
    size = max(w, h)
    cx = x + w / 2
    cy = y + h / 2
    x1 = cx - size / 2
    y1 = cy - size / 2
    x2 = x1 + size
    y2 = y1 + size

    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)

    return int(x1), int(y1), int(x2), int(y2)

def draw_bbox(image, bbox, color=(0, 255, 0), thickness=3):
    """Draw bounding box on image. bbox is [x, y, w, h] format."""
    img = image.copy()
    x, y, w, h = [int(v) for v in bbox]

    # Draw rectangle edges
    img[y:y+thickness, x:x+w] = color  # top
    img[y+h-thickness:y+h, x:x+w] = color  # bottom
    img[y:y+h, x:x+thickness] = color  # left
    img[y:y+h, x+w-thickness:x+w] = color  # right

    return img

def get_mask_border(mask, thickness=2):
    """Get the border/contour of a binary mask."""
    from scipy import ndimage
    mask_bool = mask > 0
    # Dilate and erode to get border
    dilated = ndimage.binary_dilation(mask_bool, iterations=thickness)
    eroded = ndimage.binary_erosion(mask_bool, iterations=1)
    border = dilated & ~eroded
    return border

def apply_mask_overlay(image, mask, fill_color=(100, 150, 255), border_color=(255, 255, 255), alpha=0.4, border_thickness=3):
    """Apply semi-transparent mask overlay with border on image."""
    img = image.copy().astype(np.float32)
    mask_bool = mask > 0

    # Apply fill color
    for c in range(3):
        img[:, :, c] = np.where(
            mask_bool,
            img[:, :, c] * (1 - alpha) + fill_color[c] * alpha,
            img[:, :, c]
        )

    # Apply white border
    border = get_mask_border(mask, thickness=border_thickness)
    for c in range(3):
        img[:, :, c] = np.where(
            border,
            border_color[c],
            img[:, :, c]
        )

    return img.astype(np.uint8)

def visualize_sample(coco, ann, output_prefix):
    """Generate visualizations for a single annotation."""

    # Load image
    img_info = coco.loadImgs(ann["image_id"])[0]
    file_name = get_file_name(img_info)
    img_path = os.path.join(IMG_ROOT, file_name)
    pil_image = Image.open(img_path).convert("RGB")
    image = np.array(pil_image)
    img_height, img_width = image.shape[:2]

    # Get mask and bbox
    mask = coco.annToMask(ann)
    bbox = ann["bbox"]  # [x, y, w, h]

    # 1. Normal image
    img_normal = image.copy()

    # 2. Normal + bbox
    img_bbox = draw_bbox(image.copy(), bbox, color=(0, 255, 0), thickness=4)

    # 3. Normal + mask (blue fill with white border)
    img_mask = apply_mask_overlay(image.copy(), mask, fill_color=(70, 130, 255), alpha=0.5)

    # 4. Normal + mask + bbox
    img_mask_bbox = apply_mask_overlay(image.copy(), mask, fill_color=(70, 130, 255), alpha=0.5)
    img_mask_bbox = draw_bbox(img_mask_bbox, bbox, color=(0, 255, 0), thickness=4)

    # 5. Cropped image with 10% padding (square)
    x1, y1, x2, y2 = pad_bbox_to_square(bbox, img_width, img_height, padding=0.1)
    img_cropped = pil_image.crop((x1, y1, x2, y2))
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_cropped = mask_pil.crop((x1, y1, x2, y2))

    # 6. Cropped image resized to 96x96
    img_cropped_96 = img_cropped.resize((96, 96), Image.BILINEAR)
    mask_cropped_96 = mask_cropped.resize((96, 96), Image.NEAREST)

    # Apply mask overlay to cropped versions (blue fill with white border)
    img_cropped_np = np.array(img_cropped)
    mask_cropped_np = np.array(mask_cropped)
    img_cropped_mask = apply_mask_overlay(img_cropped_np, mask_cropped_np, fill_color=(70, 130, 255), alpha=0.5)

    img_cropped_96_np = np.array(img_cropped_96)
    mask_cropped_96_np = np.array(mask_cropped_96)
    img_cropped_96_mask = apply_mask_overlay(img_cropped_96_np, mask_cropped_96_np, fill_color=(70, 130, 255), alpha=0.5, border_thickness=1)

    # Save individual images
    Image.fromarray(img_normal).save(os.path.join(OUTPUT_DIR, f"{output_prefix}_1_normal.png"))
    Image.fromarray(img_bbox).save(os.path.join(OUTPUT_DIR, f"{output_prefix}_2_bbox.png"))
    Image.fromarray(img_mask).save(os.path.join(OUTPUT_DIR, f"{output_prefix}_3_mask.png"))
    Image.fromarray(img_mask_bbox).save(os.path.join(OUTPUT_DIR, f"{output_prefix}_4_mask_bbox.png"))
    img_cropped.save(os.path.join(OUTPUT_DIR, f"{output_prefix}_5_cropped.png"))
    Image.fromarray(img_cropped_mask).save(os.path.join(OUTPUT_DIR, f"{output_prefix}_6_cropped_mask.png"))
    img_cropped_96.save(os.path.join(OUTPUT_DIR, f"{output_prefix}_7_cropped_96.png"))
    Image.fromarray(img_cropped_96_mask).save(os.path.join(OUTPUT_DIR, f"{output_prefix}_8_cropped_96_mask.png"))

    # Create combined figure (2x4 grid)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(img_normal)
    axes[0, 0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_bbox)
    axes[0, 1].set_title("+ Bounding Box", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(img_mask)
    axes[0, 2].set_title("+ Mask", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(img_mask_bbox)
    axes[0, 3].set_title("+ Mask + BBox", fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')

    axes[1, 0].imshow(img_cropped)
    axes[1, 0].set_title("Cropped (10% pad)", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img_cropped_mask)
    axes[1, 1].set_title("Cropped + Mask", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(img_cropped_96)
    axes[1, 2].set_title("96×96", fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(img_cropped_96_mask)
    axes[1, 3].set_title("96×96 + Mask", fontsize=12, fontweight='bold')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_combined.png"), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_combined.pdf"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualizations to {OUTPUT_DIR}/{output_prefix}_*.png")

def get_file_name(img_info):
    """Get file name from image info, handling COCO and LVIS formats."""
    if "file_name" in img_info:
        return img_info["file_name"]
    elif "coco_url" in img_info:
        return img_info["coco_url"].split("/")[-1]
    else:
        return f"{img_info['id']:012d}.jpg"

def get_valid_annotations(coco, img_root, min_area=5000):
    """Get annotations with valid segmentation and existing images."""
    # Build set of existing images
    existing_images = set()
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        file_name = get_file_name(img_info)
        img_path = os.path.join(img_root, file_name)
        if os.path.exists(img_path):
            existing_images.add(img_id)

    # Filter annotations
    all_anns = [
        ann for ann in coco.loadAnns(coco.getAnnIds())
        if "segmentation" in ann
        and ann.get("iscrowd", 0) == 0
        and ann["area"] > min_area
        and ann["image_id"] in existing_images
    ]
    return all_anns

def sample_and_visualize(coco, all_anns, dataset_name, num_samples=5):
    """Sample and visualize annotations from a dataset."""
    for i in range(num_samples):
        sample_ann = random.choice(all_anns)
        cat_info = coco.loadCats(sample_ann["category_id"])[0]
        cat_name = cat_info["name"]
        print(f"{dataset_name} Sample {i+1}: id={sample_ann['id']}, category={cat_name}, area={sample_ann['area']:.0f}")
        visualize_sample(coco, sample_ann, f"{dataset_name.lower()}_sample_{i+1}_{cat_name}")

def main():
    random.seed(420)

    # === COCO Validation ===
    # print("\n" + "="*50)
    # print("Loading COCO validation annotations...")
    # print("="*50)
    # coco = COCO(COCO_ANN_FILE)
    # coco_anns = get_valid_annotations(coco, IMG_ROOT)
    # print(f"Found {len(coco_anns)} valid COCO annotations")
    # sample_and_visualize(coco, coco_anns, "COCO", num_samples=100)

    # === LVIS Validation ===
    print("\n" + "="*50)
    print("Loading LVIS validation annotations...")
    print("="*50)
    lvis = COCO(LVIS_ANN_FILE)  # LVIS format is compatible with COCO API
    lvis_anns = get_valid_annotations(lvis, IMG_ROOT)
    print(f"Found {len(lvis_anns)} valid LVIS annotations")
    sample_and_visualize(lvis, lvis_anns, "LVIS", num_samples=100)

if __name__ == "__main__":
    main()
