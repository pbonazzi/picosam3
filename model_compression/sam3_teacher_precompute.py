import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
import wandb

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from .utils import pad_bbox_to_square


BASE_DIR = "/datasets/pbonazzi/picosam3_data"
IMG_ROOT = os.path.join(BASE_DIR, "train2017")
ANN_FILE = os.path.join(BASE_DIR, "annotations/instances_train2017.json")

IMAGE_SIZE = 96
CACHE_DIR = os.path.join(BASE_DIR, "teacher_sam3_logits")
os.makedirs(CACHE_DIR, exist_ok=True)

MIN_ROI_SIZE = 4
VIS_INTERVAL = 10000  # Log visualization every N processed annotations


def cache_path_for_ann(ann_id: int):
    return os.path.join(CACHE_DIR, f"ann_{ann_id}.pt")


def visualize_wandb(image, roi, teacher_mask, ann_id, score):
    """Visualize teacher mask overlaid on image.

    Args:
        image: Full image as numpy array (H, W, 3)
        roi: Region of interest (rx1, ry1, rx2, ry2)
        teacher_mask: Mask probabilities (IMAGE_SIZE, IMAGE_SIZE), values 0-1
        ann_id: Annotation ID for caption
        score: Teacher model confidence score
    """
    rx1, ry1, rx2, ry2 = roi
    roi_h, roi_w = ry2 - ry1, rx2 - rx1

    # Resize mask to ROI size
    teacher_mask_t = torch.tensor(teacher_mask).unsqueeze(0).unsqueeze(0)
    teacher_mask_resized = F.interpolate(
        teacher_mask_t,
        size=(roi_h, roi_w),
        mode="bilinear",
        align_corners=False
    ).squeeze().numpy()

    # Create overlay - green channel for mask, red for ROI boundary
    overlay = image.copy().astype(np.float32)

    # Apply green mask overlay (only where mask > 0.5)
    mask_binary = teacher_mask_resized > 0.5
    overlay[ry1:ry2, rx1:rx2, 1] = np.where(
        mask_binary,
        np.clip(overlay[ry1:ry2, rx1:rx2, 1] + 100, 0, 255),
        overlay[ry1:ry2, rx1:rx2, 1]
    )

    # Draw ROI boundary in red
    overlay[ry1:ry1+2, rx1:rx2, 0] = 255  # top
    overlay[ry2-2:ry2, rx1:rx2, 0] = 255  # bottom
    overlay[ry1:ry2, rx1:rx1+2, 0] = 255  # left
    overlay[ry1:ry2, rx2-2:rx2, 0] = 255  # right

    overlay = overlay.astype(np.uint8)

    wandb.log({
        "teacher_visualization": wandb.Image(
            overlay,
            caption=f"ann_id={ann_id}, score={score:.3f}"
        ),
        "mask_only": wandb.Image(
            (teacher_mask * 255).astype(np.uint8),
            caption=f"mask ann_id={ann_id}"
        )
    })


def bbox_to_cxcywh_normalized(bbox, img_w, img_h):
    """Convert COCO bbox [x, y, w, h] to normalized [cx, cy, w, h] format."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return [cx, cy, nw, nh]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    wandb.init(
        project="PicoSAM3-teacher-cache",
        name=f"sam3_binary_mask_teacher_rank{args.rank}",
        config={
            "image_size": IMAGE_SIZE,
            "teacher": "sam3",
            "mask_type": "binary_image_space",
            "rank": args.rank,
            "world_size": args.world_size,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build SAM3 model
    model = build_sam3_image_model(
        device="cuda",
        eval_mode=True,
        load_from_HF=True,
        enable_segmentation=True,
    )

    # Create processor for inference
    processor = Sam3Processor(
        model=model,
        device=device,
        confidence_threshold=0.0,  # We want all predictions
    )

    coco = COCO(ANN_FILE)

    anns = [
        ann for ann in coco.loadAnns(coco.getAnnIds())
        if "segmentation" in ann and ann.get("iscrowd", 0) == 0
    ]

    anns_by_img = {}
    for ann in anns:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # Shard images across workers
    all_image_ids = sorted(anns_by_img.keys())
    shard_ids = all_image_ids[args.rank::args.world_size]
    anns_by_img = {img_id: anns_by_img[img_id] for img_id in shard_ids}

    ann_counter = 0
    processed_counter = 0
    skipped_cached = 0
    skipped_small = 0
    skipped_empty = 0

    print(f"Total annotations to process: {len(anns)}")
    print(f"Total images: {len(anns_by_img)}")

    wandb.log({
        "total_annotations": len(anns),
        "total_images": len(anns_by_img),
        "status": "started"
    })

    with torch.no_grad():
        for image_id, ann_list in tqdm(
            anns_by_img.items(),
            desc="Caching teacher masks"
        ):
            # Filter out already cached annotations
            anns_to_process = []
            for ann in ann_list:
                out_path = cache_path_for_ann(ann["id"])
                if os.path.exists(out_path):
                    skipped_cached += 1
                    if skipped_cached % 1000 == 0:
                        print(f"Skipped {skipped_cached} cached annotations so far...")
                        wandb.log({"skipped_cached": skipped_cached})
                else:
                    anns_to_process.append(ann)

            if not anns_to_process:
                continue

            img_info = coco.loadImgs(image_id)[0]
            img_path = os.path.join(
                IMG_ROOT,
                img_info.get("file_name", f"{image_id:012d}.jpg")
            )

            image = Image.open(img_path).convert("RGB")
            img_np = np.array(image)
            img_h, img_w = img_np.shape[:2]

            # Set image once per image
            state = processor.set_image(image)

            # Process each annotation individually
            for ann in anns_to_process:
                # Convert bbox to normalized cxcywh format
                box_normalized = bbox_to_cxcywh_normalized(ann["bbox"], img_w, img_h)

                # Reset prompts and add geometric prompt for this box
                processor.reset_all_prompts(state)
                state = processor.add_geometric_prompt(
                    box=box_normalized,
                    label=True,  # positive box
                    state=state
                )

                # Check if we got any masks
                if "masks" not in state or len(state["masks"]) == 0:
                    skipped_empty += 1
                    continue

                # Get the mask with highest score
                # Note: masks_logits is actually probabilities (after sigmoid) despite the name
                out_scores = state["scores"].cpu().numpy()
                best_idx = out_scores.argmax()
                teacher_prob_full = state["masks_logits"][best_idx, 0].cpu().numpy()
                teacher_score = float(out_scores[best_idx])

                # Compute ROI
                rx1, ry1, rx2, ry2 = pad_bbox_to_square(
                    ann["bbox"], img_w, img_h, padding=0.1
                )

                if (rx2 - rx1) < MIN_ROI_SIZE or (ry2 - ry1) < MIN_ROI_SIZE:
                    skipped_small += 1
                    continue

                roi_mask = teacher_prob_full[ry1:ry2, rx1:rx2]

                if roi_mask.size == 0 or roi_mask.shape[0] == 0 or roi_mask.shape[1] == 0:
                    skipped_empty += 1
                    continue

                # Resize ROI mask to target size
                roi_mask_t = torch.tensor(roi_mask).unsqueeze(0).unsqueeze(0)
                roi_mask_t = F.interpolate(
                    roi_mask_t,
                    size=(IMAGE_SIZE, IMAGE_SIZE),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)

                # Convert back to logits
                eps = 1e-7
                roi_mask_clamped = torch.clamp(roi_mask_t, eps, 1 - eps)
                roi_logits = torch.log(roi_mask_clamped / (1 - roi_mask_clamped))

                out_path = cache_path_for_ann(ann["id"])
                payload = {
                    "logits": roi_logits.to(torch.float16).cpu(),
                    "score": teacher_score,
                    "bbox": ann["bbox"],
                    "roi": (rx1, ry1, rx2, ry2),
                    "image_id": int(image_id),
                }

                torch.save(payload, out_path)

                processed_counter += 1
                ann_counter += 1

                # Visualization - log first 10, then every VIS_INTERVAL
                should_visualize = (
                    processed_counter <= 10 or
                    processed_counter % VIS_INTERVAL == 0
                )
                if should_visualize:
                    print(f"Creating visualization #{processed_counter} (ann_id={ann['id']}, score={teacher_score:.3f})")
                    roi_mask_vis = torch.sigmoid(roi_logits.squeeze()).cpu().numpy()
                    visualize_wandb(
                        image=img_np,
                        roi=(rx1, ry1, rx2, ry2),
                        teacher_mask=roi_mask_vis,
                        ann_id=ann["id"],
                        score=teacher_score
                    )

            # Logging per image
            total_checked = ann_counter + skipped_cached + skipped_small + skipped_empty
            if total_checked % 500 == 0:
                wandb.log({
                    "processed": processed_counter,
                    "skipped_cached": skipped_cached,
                    "skipped_small": skipped_small,
                    "skipped_empty": skipped_empty,
                    "total_checked": total_checked,
                    "progress_pct": (total_checked / len(anns)) * 100,
                })
                print(f"Progress: {processed_counter} processed, {total_checked}/{len(anns)} checked ({total_checked/len(anns)*100:.1f}%)")

    wandb.log({
        "final_processed": processed_counter,
        "final_skipped_cached": skipped_cached,
        "final_skipped_small": skipped_small,
        "final_skipped_empty": skipped_empty,
        "final_total": processed_counter + skipped_cached + skipped_small + skipped_empty,
    })

    print(f"\n=== Teacher Precompute Summary ===")
    print(f"Processed: {processed_counter}")
    print(f"Skipped (cached): {skipped_cached}")
    print(f"Skipped (too small): {skipped_small}")
    print(f"Skipped (empty ROI): {skipped_empty}")
    print(f"Total: {processed_counter + skipped_cached + skipped_small + skipped_empty}")

    wandb.finish()


if __name__ == "__main__":
    main()
