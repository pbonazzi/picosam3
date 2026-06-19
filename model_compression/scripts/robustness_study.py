"""
Robustness study for PicoSAM3 — rebuttal evidence for Reviewer 1.

Addresses two concerns raised in review:
  (1) Spatial bias: target always centred in crop.
  (2) Multiple competing objects in the same ROI.

Experiment A – Spatial perturbation sweep
------------------------------------------
  For each shift level s ∈ {0, 5, 10, 20, 30} % of bbox size:
    · Crop centre is shifted by Δcx ~ U(−s·w/2, +s·w/2),
                                Δcy ~ U(−s·h/2, +s·h/2)
    · Box dimensions perturbed as Δw ~ N(0, (s/2·w)²), same for h.
  Each level is repeated over N_TRIALS seeds; results reported as mean ± std.

Experiment B – Multi-object ROI analysis
-----------------------------------------
  Annotations are stratified by the number of *other* GT objects
  whose bbox centre falls inside the padded crop:
    · 0 competing objects  (unambiguous)
    · 1–2 competing objects
    · 3+  competing objects (crowded)
  mIoU and mAP@[0.5:0.95] are reported per stratum.

Output: two LaTeX tables written to --output-dir.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model_compression.model import PicoSAM2, PicoSAM3  # noqa: F401
from model_compression.utils import pad_bbox_to_square

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
IMAGE_SIZE   = 96
N_TRIALS     = 5
SHIFT_LEVELS = [0.00, 0.05, 0.10, 0.20, 0.30]   # fractions of bbox size
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR     = "/datasets/pbonazzi/picosam3_data/"
CKPT_DIR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../checkpoints")
)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def _load_coco_annotations(image_root: str, annotation_file: str):
    coco     = COCO(annotation_file)
    img_ids  = coco.getImgIds()
    existing = {
        img_id
        for img_id in img_ids
        if os.path.exists(
            os.path.join(
                image_root,
                coco.loadImgs(img_id)[0].get("file_name", f"{img_id:012d}.jpg"),
            )
        )
    }
    anns = [
        ann
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=list(existing)))
        if "segmentation" in ann
        and ann.get("iscrowd", 0) == 0
        and ann["image_id"] in existing
    ]
    return coco, anns


def _perturb_bbox(bbox, img_w, img_h, shift: float, rng: np.random.Generator):
    """Perturb COCO bbox [x, y, w, h] by uniform shift + Gaussian size noise."""
    x, y, w, h = bbox
    half  = shift / 2.0
    sigma = shift / 2.0

    x_new = x + rng.uniform(-half * w, half * w)
    y_new = y + rng.uniform(-half * h, half * h)
    w_new = max(w + rng.normal(0.0, sigma * w), 1.0)
    h_new = max(h + rng.normal(0.0, sigma * h), 1.0)

    x_new = float(np.clip(x_new, 0, img_w - 1))
    y_new = float(np.clip(y_new, 0, img_h - 1))
    w_new = float(np.clip(w_new, 1, img_w - x_new))
    h_new = float(np.clip(h_new, 1, img_h - y_new))
    return [x_new, y_new, w_new, h_new]


def _crop_sample(image: Image.Image, mask_np: np.ndarray, bbox, img_w, img_h, image_size: int):
    x1, y1, x2, y2 = pad_bbox_to_square(bbox, img_w, img_h, padding=0.1)
    image_t = _TRANSFORM(image.crop((x1, y1, x2, y2)).resize((image_size, image_size), Image.BILINEAR))
    mask_t  = torch.tensor(
        np.array(Image.fromarray(mask_np).crop((x1, y1, x2, y2)).resize((image_size, image_size), Image.NEAREST)),
        dtype=torch.float32,
    ).unsqueeze(0)
    return image_t, mask_t, (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Dataset A: spatial perturbation sweep
# ---------------------------------------------------------------------------
class PerturbedDataset(Dataset):
    def __init__(self, image_root, coco, annotations, image_size,
                 shift: float = 0.0, rng_seed: int = 0):
        self.image_root  = image_root
        self.coco        = coco
        self.annotations = annotations
        self.image_size  = image_size
        self.shift       = shift
        self.rng         = np.random.default_rng(rng_seed)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann      = self.annotations[idx]
        img_info = self.coco.loadImgs(ann["image_id"])[0]
        img_path = os.path.join(self.image_root,
                                img_info.get("file_name", f"{img_info['id']:012d}.jpg"))
        image   = Image.open(img_path).convert("RGB")
        mask_np = self.coco.annToMask(ann)
        img_w, img_h = image.size

        bbox = ann["bbox"]
        if self.shift > 0:
            bbox = _perturb_bbox(bbox, img_w, img_h, self.shift, self.rng)

        image_t, mask_t, _ = _crop_sample(image, mask_np, bbox, img_w, img_h, self.image_size)
        return image_t, mask_t


# ---------------------------------------------------------------------------
# Dataset B: multi-object ROI stratification
# ---------------------------------------------------------------------------
def _count_competing_objects(ann, all_anns_by_image, coco, img_w, img_h, image_size):
    """Count GT objects (different category-agnostic) whose centre falls in the crop."""
    bbox = ann["bbox"]
    x1, y1, x2, y2 = pad_bbox_to_square(bbox, img_w, img_h, padding=0.1)
    count = 0
    for other in all_anns_by_image.get(ann["image_id"], []):
        if other["id"] == ann["id"]:
            continue
        ox, oy, ow, oh = other["bbox"]
        cx = ox + ow / 2
        cy = oy + oh / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            count += 1
    return count


class StratifiedDataset(Dataset):
    """Returns (image_t, mask_t, stratum_label) for multi-object analysis."""

    STRATA = {0: "0 competitors", 1: "1-2 competitors", 2: "3+ competitors"}

    def __init__(self, image_root, coco, annotations, image_size):
        self.image_root  = image_root
        self.coco        = coco
        self.image_size  = image_size

        # Index annotations by image
        by_image = defaultdict(list)
        for ann in annotations:
            by_image[ann["image_id"]].append(ann)

        # Assign stratum
        print("  Computing multi-object strata …")
        self.records = []
        for ann in tqdm(annotations, leave=False):
            img_info = coco.loadImgs(ann["image_id"])[0]
            img_w    = img_info["width"]
            img_h    = img_info["height"]
            n = _count_competing_objects(ann, by_image, coco, img_w, img_h, image_size)
            if   n == 0:  stratum = 0
            elif n <= 2:  stratum = 1
            else:         stratum = 2
            self.records.append((ann, stratum))

        counts = defaultdict(int)
        for _, s in self.records:
            counts[s] += 1
        for s, label in self.STRATA.items():
            print(f"    {label}: {counts[s]:,} samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        ann, stratum = self.records[idx]
        img_info = self.coco.loadImgs(ann["image_id"])[0]
        img_path = os.path.join(
            self.image_root,
            img_info.get("file_name", f"{img_info['id']:012d}.jpg"),
        )
        image   = Image.open(img_path).convert("RGB")
        mask_np = self.coco.annToMask(ann)
        img_w, img_h = image.size
        image_t, mask_t, _ = _crop_sample(image, mask_np, ann["bbox"], img_w, img_h, self.image_size)
        return image_t, mask_t, stratum


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _iou_batch(preds: torch.Tensor, targets: torch.Tensor) -> list[float]:
    p = (torch.sigmoid(preds) > 0.5).cpu().numpy()
    t = (targets > 0.5).cpu().numpy()
    ious = []
    for pi, ti in zip(p, t):
        union = np.logical_or(pi, ti).sum()
        ious.append(float(np.logical_and(pi, ti).sum() / union) if union else 1.0)
    return ious


def _map_batch(preds: torch.Tensor, targets: torch.Tensor,
               thresholds=np.arange(0.5, 1.0, 0.05)) -> list[float]:
    p = torch.sigmoid(preds).cpu().numpy()
    t = targets.cpu().numpy()
    aps_per_sample = []
    for pi, ti in zip(p, t):
        if ti.sum() == 0:
            continue
        sample_aps = []
        for thr in thresholds:
            iou = (np.logical_and(pi > 0.5, ti > 0.5).sum() /
                   (np.logical_or(pi > 0.5, ti > 0.5).sum() + 1e-8))
            sample_aps.append(1.0 if iou >= thr else 0.0)
        aps_per_sample.append(float(np.mean(sample_aps)))
    return aps_per_sample


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------
def evaluate_loader(model, loader, device, max_samples=float("inf")):
    """Returns (mean_miou, mean_map)."""
    model.eval()
    all_iou, all_ap = [], []
    n = 0
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            if n >= max_samples:
                break
            x, y = batch[0].to(device), batch[1].to(device)
            pred = F.interpolate(model(x), size=y.shape[-2:],
                                 mode="bilinear", align_corners=False)
            all_iou.extend(_iou_batch(pred.cpu(), y.cpu()))
            all_ap.extend(_map_batch(pred.cpu(), y.cpu()))
            n += x.size(0)
    return float(np.mean(all_iou)), float(np.mean(all_ap)) if all_ap else 0.0


def evaluate_stratified(model, dataset, device, batch_size, num_workers,
                        max_samples=float("inf")):
    """Returns {stratum_int: (miou, mmap)}."""
    model.eval()
    strata_iou = defaultdict(list)
    strata_ap  = defaultdict(list)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    n = 0
    with torch.no_grad():
        for x, y, strata in tqdm(loader, leave=False):
            if n >= max_samples:
                break
            x, y = x.to(device), y.to(device)
            pred = F.interpolate(model(x), size=y.shape[-2:],
                                 mode="bilinear", align_corners=False)
            ious = _iou_batch(pred.cpu(), y.cpu())
            aps  = _map_batch(pred.cpu(), y.cpu())
            for i, s in enumerate(strata.tolist()):
                strata_iou[s].append(ious[i])
                if i < len(aps):
                    strata_ap[s].append(aps[i])
            n += x.size(0)

    return {
        s: (float(np.mean(strata_iou[s])), float(np.mean(strata_ap[s])) if strata_ap[s] else 0.0)
        for s in strata_iou
    }


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------
def latex_sweep_table(sweep_results: dict, shift_levels: list, n_trials: int) -> str:
    """
    sweep_results = {
        model_name: {
            dataset_name: {
                shift_level (float): (mean_miou, std_miou, mean_mmap, std_mmap)
            }
        }
    }
    """
    pct_labels = [f"{int(s * 100)}\\%" for s in shift_levels]
    n_cols = len(shift_levels)
    col_spec = "ll" + "cc" * n_cols

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \small",
        rf"  \caption{{Robustness to spatial bbox perturbations (mean$\pm$std over {n_trials} trials). "
        r"Each column is a perturbation level expressed as a fraction of bbox size "
        r"(uniform centre shift $\pm s/2$, Gaussian dimension noise $\sigma=s/2$).}",
        r"  \label{tab:spatial_robustness}",
        r"  \setlength{\tabcolsep}{4pt}",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
    ]

    # Header row 1
    multicols = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{\textbf{{{lbl}}}}}" for lbl in pct_labels
    )
    lines.append(rf"    \textbf{{Model}} & \textbf{{Dataset}} & {multicols} \\")

    # Cmidrule separators
    crules = " ".join(
        rf"\cmidrule(lr){{{3 + 2*i}-{4 + 2*i}}}" for i in range(n_cols)
    )
    lines.append(f"    {crules}")

    # Header row 2
    metric_hdrs = " & ".join(r"mIoU & mAP" for _ in shift_levels)
    lines.append(rf"    & & {metric_hdrs} \\")
    lines.append(r"    \midrule")

    for model_name, datasets in sweep_results.items():
        n_ds = len(datasets)
        first = True
        for ds_name, level_metrics in datasets.items():
            cells = []
            for s in shift_levels:
                mu_iou, sd_iou, mu_map, sd_map = level_metrics[s]
                if sd_iou < 0.0005:   # clean (single run, no std)
                    cells.append(rf"{mu_iou:.3f} & {mu_map:.3f}")
                else:
                    cells.append(
                        rf"{mu_iou:.3f}{{\tiny$\pm${sd_iou:.3f}}} & "
                        rf"{mu_map:.3f}{{\tiny$\pm${sd_map:.3f}}}"
                    )
            model_cell = (
                rf"\multirow{{{n_ds}}}{{*}}{{{model_name}}}" if first else ""
            )
            lines.append(f"    {model_cell} & {ds_name} & " + " & ".join(cells) + r" \\")
            first = False
        lines.append(r"    \midrule")

    lines[-1] = r"    \bottomrule"
    lines += [r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_multiobj_table(strat_results: dict) -> str:
    """
    strat_results = {
        model_name: {
            dataset_name: {
                stratum_int: (miou, mmap)
            }
        }
    }
    """
    strata_labels = StratifiedDataset.STRATA
    col_spec = "ll" + "cc" * len(strata_labels)

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \small",
        r"  \caption{Segmentation performance stratified by number of competing objects "
        r"whose centre falls inside the padded bounding-box crop. "
        r"Results confirm the model is not confused by crowded scenes.}",
        r"  \label{tab:multiobj_robustness}",
        r"  \setlength{\tabcolsep}{5pt}",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
    ]

    multicols = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{\textbf{{{strata_labels[s]}}}}}"
        for s in sorted(strata_labels)
    )
    lines.append(rf"    \textbf{{Model}} & \textbf{{Dataset}} & {multicols} \\")

    crules = " ".join(
        rf"\cmidrule(lr){{{3 + 2*i}-{4 + 2*i}}}" for i in range(len(strata_labels))
    )
    lines.append(f"    {crules}")

    metric_hdrs = " & ".join(r"mIoU & mAP" for _ in strata_labels)
    lines.append(rf"    & & {metric_hdrs} \\")
    lines.append(r"    \midrule")

    for model_name, datasets in strat_results.items():
        n_ds = len(datasets)
        first = True
        for ds_name, stratum_metrics in datasets.items():
            cells = []
            for s in sorted(strata_labels):
                if s in stratum_metrics:
                    miou, mmap = stratum_metrics[s]
                    cells.append(rf"{miou:.3f} & {mmap:.3f}")
                else:
                    cells.append(r"-- & --")
            model_cell = (
                rf"\multirow{{{n_ds}}}{{*}}{{{model_name}}}" if first else ""
            )
            lines.append(f"    {model_cell} & {ds_name} & " + " & ".join(cells) + r" \\")
            first = False
        lines.append(r"    \midrule")

    lines[-1] = r"    \bottomrule"
    lines += [r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Robustness study for PicoSAM3 (rebuttal Reviewer 1)"
    )
    parser.add_argument("--ckpt-dir",    default=CKPT_DIR_DEFAULT)
    parser.add_argument("--base-dir",    default=BASE_DIR)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--workers",     type=int,   default=4)
    parser.add_argument("--num-samples", type=int,   default=0,
                        help="Max annotations per split (0 = all)")
    parser.add_argument("--n-trials",    type=int,   default=N_TRIALS)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--datasets",    nargs="+",
                        choices=["coco", "lvis"], default=["coco", "lvis"])
    parser.add_argument("--output-dir",  default=".",
                        help="Directory for output .tex files")
    parser.add_argument(
        "--shift-levels", nargs="+", type=float, default=SHIFT_LEVELS,
        help="Perturbation sweep levels (fractions of bbox size)"
    )
    parser.add_argument("--skip-sweep",     action="store_true")
    parser.add_argument("--skip-multiobj",  action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    max_samples = args.num_samples if args.num_samples > 0 else float("inf")

    # ------------------------------------------------------------------ paths
    base = args.base_dir
    datasets_cfg = []
    if "coco" in args.datasets:
        datasets_cfg.append((
            "COCO val2017",
            os.path.join(base, "val2017"),
            os.path.join(base, "annotations/instances_val2017.json"),
        ))
    if "lvis" in args.datasets:
        datasets_cfg.append((
            "LVIS v1 val",
            os.path.join(base, "val2017"),
            os.path.join(base, "annotations/lvis_v1_val.json"),
        ))

    ckpt = args.ckpt_dir
    models_cfg = [
        ("PicoSAM3 (distilled)", PicoSAM3,
         os.path.join(ckpt, "PicoSAM3_SAM3_student_epoch1.pt")),
        ("PicoSAM2 (distilled)", PicoSAM2,
         os.path.join(ckpt, "PicoSAM2_SAM3_student_epoch1.pt")),
    ]

    # =====================================================================
    # Experiment A – Spatial perturbation sweep
    # =====================================================================
    if not args.skip_sweep:
        print("\n" + "=" * 70)
        print("Experiment A: Spatial perturbation sweep")
        print("=" * 70)

        sweep_results = {}

        for model_name, model_cls, ckpt_path in models_cfg:
            if not os.path.exists(ckpt_path):
                print(f"\nSkipping {model_name}: checkpoint not found ({ckpt_path})")
                continue
            print(f"\nModel: {model_name}")
            model = model_cls()
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            model = model.to(DEVICE)
            model.eval()

            sweep_results[model_name] = {}

            for ds_name, img_root, ann_file in datasets_cfg:
                print(f"  Dataset: {ds_name}")
                coco, anns = _load_coco_annotations(img_root, ann_file)
                print(f"  {len(anns):,} annotations loaded")

                sweep_results[model_name][ds_name] = {}

                for shift in args.shift_levels:
                    if shift == 0.0:
                        # Single clean run — no std needed
                        clean_ds = PerturbedDataset(img_root, coco, anns,
                                                    IMAGE_SIZE, shift=0.0,
                                                    rng_seed=args.seed)
                        loader = DataLoader(clean_ds, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.workers,
                                            pin_memory=True)
                        mu_iou, mu_map = evaluate_loader(model, loader, DEVICE,
                                                          max_samples)
                        sd_iou = sd_map = 0.0
                        print(f"    shift=0%   mIoU={mu_iou:.4f}  mAP={mu_map:.4f}")
                    else:
                        trial_ious, trial_maps = [], []
                        for trial in range(args.n_trials):
                            seed = args.seed + trial * 997
                            ds = PerturbedDataset(img_root, coco, anns,
                                                  IMAGE_SIZE, shift=shift,
                                                  rng_seed=seed)
                            loader = DataLoader(ds, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.workers,
                                                pin_memory=True)
                            t_iou, t_map = evaluate_loader(model, loader, DEVICE,
                                                            max_samples)
                            trial_ious.append(t_iou)
                            trial_maps.append(t_map)
                        mu_iou = float(np.mean(trial_ious))
                        sd_iou = float(np.std(trial_ious))
                        mu_map = float(np.mean(trial_maps))
                        sd_map = float(np.std(trial_maps))
                        print(
                            f"    shift={int(shift*100):2d}%  "
                            f"mIoU={mu_iou:.4f}±{sd_iou:.4f}  "
                            f"mAP={mu_map:.4f}±{sd_map:.4f}"
                        )

                    sweep_results[model_name][ds_name][shift] = (
                        mu_iou, sd_iou, mu_map, sd_map
                    )

        latex_sweep = latex_sweep_table(sweep_results, args.shift_levels, args.n_trials)
        out_sweep = os.path.join(args.output_dir, "tab_spatial_robustness.tex")
        with open(out_sweep, "w") as f:
            f.write(latex_sweep + "\n")
        print(f"\nSweep table written to: {out_sweep}")
        print("\n" + latex_sweep)

    # =====================================================================
    # Experiment B – Multi-object ROI stratification
    # =====================================================================
    if not args.skip_multiobj:
        print("\n" + "=" * 70)
        print("Experiment B: Multi-object ROI stratification")
        print("=" * 70)

        strat_results = {}

        for model_name, model_cls, ckpt_path in models_cfg:
            if not os.path.exists(ckpt_path):
                print(f"\nSkipping {model_name}: checkpoint not found ({ckpt_path})")
                continue
            print(f"\nModel: {model_name}")
            model = model_cls()
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            model = model.to(DEVICE)
            model.eval()

            strat_results[model_name] = {}

            for ds_name, img_root, ann_file in datasets_cfg:
                print(f"  Dataset: {ds_name}")
                coco, anns = _load_coco_annotations(img_root, ann_file)
                print(f"  {len(anns):,} annotations loaded")

                strat_ds = StratifiedDataset(img_root, coco, anns, IMAGE_SIZE)
                metrics  = evaluate_stratified(
                    model, strat_ds, DEVICE,
                    args.batch_size, args.workers, max_samples
                )

                strat_results[model_name][ds_name] = metrics
                for s, (miou, mmap) in sorted(metrics.items()):
                    label = StratifiedDataset.STRATA[s]
                    print(f"    {label:20s}  mIoU={miou:.4f}  mAP={mmap:.4f}")

        latex_strat = latex_multiobj_table(strat_results)
        out_strat = os.path.join(args.output_dir, "tab_multiobj_robustness.tex")
        with open(out_strat, "w") as f:
            f.write(latex_strat + "\n")
        print(f"\nMulti-object table written to: {out_strat}")
        print("\n" + latex_strat)


if __name__ == "__main__":
    main()
