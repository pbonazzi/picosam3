
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import itertools
import model_compression_toolkit as mct
from PIL import Image

from model_compression.dataset import PicoSAMDataset, custom_collate
from model_compression.model import PicoSAM2, PicoSAM3

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

IMAGE_SIZE = 96
NUM_SAMPLES = float("inf") # 1000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/datasets/pbonazzi/picosam3_data/"

CACHE_DIR = os.path.join(BASE_DIR, "teacher_sam3_logits")

CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
COCO_IMG_ROOT = os.path.join(BASE_DIR, "val2017")
COCO_ANN_FILE = os.path.join(BASE_DIR, "annotations/instances_val2017.json")

LVIS_IMG_ROOT = os.path.join(BASE_DIR, "val2017")  # LVIS uses COCO val2017 images
LVIS_ANN_FILE = os.path.join(BASE_DIR, "annotations/lvis_v1_val.json") 

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (tensor * std + mean).clamp(0,1)

def calc_miou(preds, targets):
    preds = (torch.sigmoid(preds) > 0.5).cpu().numpy()
    targets = (targets > 0.5).cpu().numpy()
    return np.mean([
        (np.logical_and(p,t).sum() / np.logical_or(p,t).sum()) if np.logical_or(p,t).sum() else 1.0
        for p,t in zip(preds, targets)
    ])

def calc_map_iou_range(preds, targets, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    preds = torch.sigmoid(preds).cpu().numpy()
    targets = targets.cpu().numpy()
    aps = []

    for iou_thresh in iou_thresholds:
        ap_per_thresh = []
        for p, t in zip(preds, targets):
            if t.sum() == 0: continue
            p_bin = p > 0.5
            t_bin = t > 0.5
            iou = np.logical_and(p_bin, t_bin).sum() / (np.logical_or(p_bin, t_bin).sum() + 1e-8)
            ap_per_thresh.append(1.0 if iou >= iou_thresh else 0.0)
        if ap_per_thresh:
            aps.append(np.mean(ap_per_thresh))

    return float(np.mean(aps)) if aps else 0.0

def evaluate_picosam(model, loader, name, device=None):
    if device is None:
        device = next(model.parameters()).device
    else:
        model = model.to(device)
    print(f"\nEvaluating: {name} (on {device})")
    preds, gts, mious = [], [], []
    for i, (x, y, _, _, _, _) in enumerate(tqdm(loader)):
        if i >= NUM_SAMPLES: break
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
        pred = F.interpolate(pred, size=y.shape[-2:], mode="bilinear", align_corners=False)
        preds.append(pred.cpu())
        gts.append(y.cpu())
        mious.append(calc_miou(pred.cpu(), y.cpu()))
    preds = torch.cat(preds)
    gts = torch.cat(gts)
    print(f"{name} -> mIoU: {np.mean(mious):.4f}, mAP@[0.5:0.95]: {calc_map_iou_range(preds, gts):.4f}")

def evaluate_sam2(predictor, dataset, name):
    print(f"\nEvaluating: {name}")
    preds, gts, mious = [], [], []
    for i in range(len(dataset)):
        if i >= NUM_SAMPLES: break
        image, mask, prompt_coords, _, _, _ = dataset[i]
        image_np = (unnormalize(image).permute(1,2,0).numpy() * 255).astype(np.uint8)
        predictor.set_image(image_np)
        pt = np.array([[prompt_coords]])
        lbl = np.array([[1]])
        masks, scores, _ = predictor.predict(point_coords=pt, point_labels=lbl, return_logits=False)
        best_idx = np.argmax(scores)
        pred = torch.tensor(masks[best_idx]).unsqueeze(0).unsqueeze(0)
        pred = F.interpolate(pred, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        preds.append(pred)
        gts.append(mask.unsqueeze(0))
        mious.append(calc_miou(pred, mask.unsqueeze(0)))
    preds = torch.cat(preds)
    gts = torch.cat(gts)
    print(f"{name} -> mIoU: {np.mean(mious):.4f}, mAP@[0.5:0.95]: {calc_map_iou_range(preds, gts):.4f}")

def evaluate_sam3(processor, dataset, name):
    print(f"\nEvaluating: {name}")
    preds, gts, mious = [], [], []
    for i in tqdm(range(len(dataset))):
        if i >= NUM_SAMPLES: break
        image, mask, prompt_coords, _, _, _ = dataset[i]
        # Convert to PIL Image for SAM3
        image_np = (unnormalize(image).permute(1,2,0).numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # Set image and create box prompt covering center region
        # Box format: [cx, cy, w, h] normalized
        state = processor.set_image(pil_image)
        processor.reset_all_prompts(state)
        state = processor.add_geometric_prompt(
            box=[0.5, 0.5, 0.8, 0.8],  # centered box covering 80% of image
            label=True,
            state=state
        )

        if "masks" not in state or len(state["masks"]) == 0:
            # No mask predicted, skip
            continue

        # Get best mask
        scores = state["scores"].cpu().numpy()
        best_idx = scores.argmax()
        pred_mask = state["masks"][best_idx, 0].cpu().float()
        pred = pred_mask.unsqueeze(0).unsqueeze(0)
        pred = F.interpolate(pred, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        preds.append(pred)
        gts.append(mask.unsqueeze(0))
        mious.append(calc_miou(pred, mask.unsqueeze(0)))

    if preds:
        preds = torch.cat(preds)
        gts = torch.cat(gts)
        print(f"{name} -> mIoU: {np.mean(mious):.4f}, mAP@[0.5:0.95]: {calc_map_iou_range(preds, gts):.4f}")
    else:
        print(f"{name} -> No valid predictions")

if __name__ == "__main__":
    coco_data = PicoSAMDataset(COCO_IMG_ROOT, COCO_ANN_FILE, IMAGE_SIZE, CACHE_DIR, require_cache=False)
    coco_loader = DataLoader(coco_data, batch_size=1, shuffle=False)
    lvis_data = PicoSAMDataset(LVIS_IMG_ROOT, LVIS_ANN_FILE, IMAGE_SIZE, CACHE_DIR, require_cache=False)
    lvis_loader = DataLoader(lvis_data, batch_size=1, shuffle=False)

    # PicoSAM2
    scratch = PicoSAM2().to(DEVICE)
    scratch.load_state_dict(torch.load(os.path.join(CKPT_DIR, "PicoSAM2_epoch1.pt"))); scratch.eval()
    
    distilled_from_sam2 = PicoSAM2().to(DEVICE)
    distilled_from_sam2.load_state_dict(torch.load(os.path.join(CKPT_DIR, "PicoSAM2_student_epoch1.pt"))); distilled_from_sam2.eval()
    
    distilled_from_sam3 = PicoSAM2().to(DEVICE)
    distilled_from_sam3.load_state_dict(torch.load(os.path.join(CKPT_DIR, "PicoSAM2_SAM3_student_epoch1.pt"))); distilled_from_sam3.eval()
    
    quant_picosam2 = PicoSAM2()
    quant_picosam2.load_state_dict(torch.load(os.path.join(CKPT_DIR, "PicoSAM2_SAM3_student_epoch1.pt"), map_location="cpu")); quant_picosam2.eval()

    # PicoSAM3

    distilled_picosam3 = PicoSAM3().to(DEVICE)
    distilled_picosam3.load_state_dict(torch.load(os.path.join(CKPT_DIR, "PicoSAM3_SAM3_student_epoch1.pt"))); distilled_picosam3.eval()

    quant_picosam3 = PicoSAM3()
    quant_picosam3.load_state_dict(torch.load(os.path.join(CKPT_DIR, "PicoSAM3_SAM3_student_epoch1.pt"), map_location="cpu")); quant_picosam3.eval() 

    def repr_dataset():
        val_iter = itertools.cycle(coco_loader)
        def generator():
            for _ in range(10):
                yield [next(val_iter)[0].to(DEVICE)]
        return generator

    tpc = mct.get_target_platform_capabilities("pytorch", "imx500")
    quant_picosam3_quantized, _ = mct.ptq.pytorch_post_training_quantization(
        quant_picosam3.to(DEVICE),
        representative_data_gen=repr_dataset(),
        target_platform_capabilities=tpc
    )
    # Keep on CUDA - MCT quantizer scales are created on the device where quantization runs

    quant_picosam2_quantized, _ = mct.ptq.pytorch_post_training_quantization(
        quant_picosam2.to(DEVICE),
        representative_data_gen=repr_dataset(),
        target_platform_capabilities=tpc
    )
    # Keep on CUDA - MCT quantizer scales are created on the device where quantization runs


    sam_variants = {
        "SAM2.1 Large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
        "SAM2.1 Base+": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
        "SAM2.1 Small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
        "SAM2.1 Tiny":  ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt")
    }
    sam_predictors = {name: SAM2ImagePredictor(build_sam2(cfg, os.path.join(CKPT_DIR, ckpt), device=DEVICE, mode="eval")) for name, (cfg, ckpt) in sam_variants.items()}

    # Evaluate PicoSAM2 variants on COCO
    #evaluate_picosam(scratch, coco_loader, "PicoSAM2 Scratch (COCO)")
    #evaluate_picosam(distilled_from_sam2, coco_loader, "PicoSAM2 Distilled from SAM2 (COCO)")
    #evaluate_picosam(distilled_from_sam3, coco_loader, "PicoSAM2 Distilled from SAM3 (COCO)")
    #evaluate_picosam(quant_picosam2_quantized, coco_loader, "PicoSAM2 Quantized (COCO)", device=DEVICE)

    # Evaluate PicoSAM3 variants on COCO
    #evaluate_picosam(distilled_picosam3, coco_loader, "PicoSAM3 Distilled from SAM3 (COCO)")
    #evaluate_picosam(quant_picosam3_quantized, coco_loader, "PicoSAM3 MCT Quantized (COCO)", device=DEVICE)

    # Evaluate PicoSAM2 variants on LVIS
    #evaluate_picosam(scratch, lvis_loader, "PicoSAM2 Scratch (LVIS)")
    #evaluate_picosam(distilled_from_sam2, lvis_loader, "PicoSAM2 Distilled from SAM2 (LVIS)")
    #evaluate_picosam(distilled_from_sam3, lvis_loader, "PicoSAM2 Distilled from SAM3 (LVIS)")
    #evaluate_picosam(quant_picosam2_quantized, lvis_loader, "PicoSAM2 Quantized (LVIS)", device="cpu")

    # Evaluate PicoSAM3 variants on LVIS
    #evaluate_picosam(distilled_picosam3, lvis_loader, "PicoSAM3 Distilled from SAM3 (LVIS)")
    evaluate_picosam(quant_picosam3_quantized, lvis_loader, "PicoSAM3 MCT Quantized (LVIS)", device=DEVICE)

    # for name, predictor in sam_predictors.items():
    #     evaluate_sam2(predictor, coco_data, f"{name} (COCO)")
    #     evaluate_sam2(predictor, lvis_data, f"{name} (LVIS)")

    # SAM3 evaluation
    # sam3_model = build_sam3_image_model(
    #     device="cuda",  # Must be string, not torch.device
    #     eval_mode=True,
    #     load_from_HF=True,
    #     enable_segmentation=True,
    # )
    # sam3_processor = Sam3Processor(model=sam3_model, device="cuda", confidence_threshold=0.0)
    # evaluate_sam3(sam3_processor, coco_data, "SAM3 (COCO)")
    # evaluate_sam3(sam3_processor, lvis_data, "SAM3 (LVIS)")
