#!/usr/bin/env python3
"""
Ablation study: effect of ROI padding percentage on PicoSAM3 inference performance.
Tests padding values [0.0, 0.05, 0.10, 0.15, 0.20] on COCO val2017 and LVIS v1 val.
The model is trained with p=0.10; we evaluate with different inference-time padding.

Usage:
  cd /home/pbonazzi/projects/eth_zurich/picosam3_journal/picosam3
  uv run python rebuttal_tables/padding_ablation.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_compression.model import PicoSAM3

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 96
BATCH_SIZE = 64
BASE_DIR = "/datasets/pbonazzi/picosam3_data"
CKPT = os.path.join(BASE_DIR, "checkpoints/PicoSAM3_SAM3_student_epoch1.pt")

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

PADDING_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20]

DATASETS = [
    ("COCO val2017",
     os.path.join(BASE_DIR, "val2017"),
     os.path.join(BASE_DIR, "annotations/instances_val2017.json")),
    ("LVIS v1 val",
     os.path.join(BASE_DIR, "val2017"),
     os.path.join(BASE_DIR, "annotations/lvis_v1_val.json")),
]


def pad_bbox_to_square(bbox, img_w, img_h, padding=0.1):
    x, y, w, h = bbox
    x -= w * padding;  y -= h * padding
    w += 2 * w * padding;  h += 2 * h * padding
    size = max(w, h)
    cx = x + w / 2;  cy = y + h / 2
    x1 = cx - size / 2;  y1 = cy - size / 2
    x2 = x1 + size;      y2 = y1 + size
    x1 = max(0, x1);     y1 = max(0, y1)
    x2 = min(img_w, x2); y2 = min(img_h, y2)
    return int(x1), int(y1), int(x2), int(y2)


class PaddingDataset(Dataset):
    def __init__(self, img_root, ann_file, padding):
        self.coco    = COCO(ann_file)
        self.img_dir = img_root
        self.padding = padding
        img_ids = self.coco.getImgIds()
        existing = {i for i in img_ids
                    if os.path.exists(os.path.join(img_root,
                        self.coco.loadImgs(i)[0].get("file_name", f"{i:012d}.jpg")))}
        self.anns = [a for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=list(existing)))
                     if "segmentation" in a and not a.get("iscrowd", 0) and a["image_id"] in existing]
        print(f"  padding={padding:.2f}: {len(self.anns):,} annotations")

    def __len__(self): return len(self.anns)

    def __getitem__(self, idx):
        ann      = self.anns[idx]
        img_info = self.coco.loadImgs(ann["image_id"])[0]
        img_path = os.path.join(self.img_dir,
                    img_info.get("file_name", f"{img_info['id']:012d}.jpg"))
        image   = Image.open(img_path).convert("RGB")
        mask_np = self.coco.annToMask(ann)
        w, h    = image.size

        x1, y1, x2, y2 = pad_bbox_to_square(ann["bbox"], w, h, self.padding)
        crop_img  = image.crop((x1, y1, x2, y2)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        crop_mask = Image.fromarray(mask_np).crop((x1, y1, x2, y2)).resize(
                        (IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        return (_TRANSFORM(crop_img),
                torch.tensor(np.array(crop_mask), dtype=torch.float32).unsqueeze(0))


def evaluate(model, ds_name, img_root, ann_file, padding):
    ds     = PaddingDataset(img_root, ann_file, padding)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)
    model.eval()
    ious, aps = [], []
    iou_thr = np.arange(0.5, 1.0, 0.05)
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False, desc=f"{ds_name} p={padding:.2f}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = F.interpolate(model(x), size=y.shape[-2:], mode="bilinear", align_corners=False)
            p_np = (torch.sigmoid(pred) > 0.5).cpu().numpy()
            t_np = (y > 0.5).cpu().numpy()
            for pi, ti in zip(p_np, t_np):
                union = np.logical_or(pi, ti).sum()
                ious.append(float(np.logical_and(pi, ti).sum() / union) if union else 1.0)
            p_prob = torch.sigmoid(pred).cpu().numpy()
            for pi, ti in zip(p_prob, t_np):
                if not ti.sum(): continue
                sample_aps = []
                for thr in iou_thr:
                    iou_v = np.logical_and(pi > 0.5, ti > 0.5).sum() / (
                            np.logical_or(pi > 0.5, ti > 0.5).sum() + 1e-8)
                    sample_aps.append(1.0 if iou_v >= thr else 0.0)
                aps.append(float(np.mean(sample_aps)))
    return float(np.mean(ious)), float(np.mean(aps)) if aps else 0.0


def main():
    print(f"Device: {DEVICE}")
    model = PicoSAM3()
    ckpt  = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(DEVICE)
    print(f"Loaded checkpoint: {CKPT}")

    results = {}  # ds_name -> {padding -> (miou, map)}
    for ds_name, img_root, ann_file in DATASETS:
        results[ds_name] = {}
        for p in PADDING_VALUES:
            miou, mmap = evaluate(model, ds_name, img_root, ann_file, p)
            results[ds_name][p] = (miou, mmap)
            print(f"  {ds_name}  p={p:.2f}  mIoU={miou:.4f}  mAP={mmap:.4f}")

    # Print LaTeX table
    print("\n\n% ── LaTeX table ─────────────────────────────────────────")
    pct_labels = [f"{int(p*100)}\\%" for p in PADDING_VALUES]
    col_spec = "l" + "cc" * len(PADDING_VALUES)
    print(r"\begin{table}[ht]")
    print(r"  \centering\small")
    print(r"  \caption{Effect of ROI padding percentage on PicoSAM3 inference-time performance"
          r" (model trained with $p\!=\!10\%$).}")
    print(r"  \label{tab:padding_ablation}")
    print(r"  \setlength{\tabcolsep}{4pt}")
    print(f"  \\begin{{tabular}}{{{col_spec}}}")
    print(r"    \toprule")
    hdr = " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{lbl}}}}}" for lbl in pct_labels)
    print(f"    \\textbf{{Dataset}} & {hdr} \\\\")
    crules = " ".join(f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(PADDING_VALUES)))
    print(f"    {crules}")
    print("    & " + " & ".join("mIoU & mAP" for _ in PADDING_VALUES) + r" \\")
    print(r"    \midrule")
    for ds_name in results:
        row = []
        for p in PADDING_VALUES:
            mi, ma = results[ds_name][p]
            if p == 0.10:
                row.append(f"\\textbf{{{mi:.3f}}} & \\textbf{{{ma:.3f}}}")
            else:
                row.append(f"{mi:.3f} & {ma:.3f}")
        short_name = "COCO" if "COCO" in ds_name else "LVIS"
        print(f"    {short_name} & " + " & ".join(row) + r" \\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")

    # Print raw results for easy reference
    print("\n\nRaw results:")
    for ds_name in results:
        for p in PADDING_VALUES:
            mi, ma = results[ds_name][p]
            print(f"  {ds_name}  p={p:.2f}  mIoU={mi:.4f}  mAP={ma:.4f}")


if __name__ == "__main__":
    main()
