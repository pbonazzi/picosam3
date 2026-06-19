import os , pdb
import torch 
from torch.utils.data import DataLoader, random_split 
from tqdm import tqdm 
import wandb 
from model_compression.utils import area_loss, mse_dice_loss, bce_dice_loss, compute_iou
from model_compression.plotting import save_visualization
from model_compression.dataset import PicoSAMDataset, custom_collate
from model_compression.model import PicoSAM2, PicoSAM3

IMAGE_SIZE = 96 
BATCH_SIZE = 64
NUM_EPOCHS = 1
LEARNING_RATE = 3e-4

BASE_DIR = "/datasets/pbonazzi/picosam3_data/"
IMG_ROOT = os.path.join(BASE_DIR, "train2017")
ANN_FILE = os.path.join(BASE_DIR, "annotations/instances_train2017.json")
CACHE_DIR = os.path.join(BASE_DIR, "teacher_sam3_logits")

NAME_RUN = "PicoSAM3_SAM3"

OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints")
IMAGE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

def train():
    wandb.init(project="PicoSAM2-distillation", config={
        "img_size": IMAGE_SIZE,
        "epochs": NUM_EPOCHS,
        "lr": LEARNING_RATE,
        "prompt_embed": "DoG_RGB"
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_model = PicoSAM3().to(device)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / 1000))

    dataset = PicoSAMDataset(IMG_ROOT, ANN_FILE, IMAGE_SIZE, CACHE_DIR)
    val_size = max(1, len(dataset) // 20)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size]) 

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, collate_fn=custom_collate)

    vis_interval = max(1, len(train_loader) // 10)

    for epoch in range(NUM_EPOCHS):
        student_model.train()
        total_loss, total_iou, num_samples = 0.0, 0.0, 0

        for batch_idx, (images_clean, gt_masks, prompts, _, teacher_logits_batch, teacher_scores_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            images_clean = images_clean.to(device, non_blocking=True)
            gt_masks = gt_masks.to(device, non_blocking=True)
            prompts = prompts.to(device, non_blocking=True)
            teacher_logits_batch = teacher_logits_batch.to(device, non_blocking=True)
            teacher_scores_batch = teacher_scores_batch.to(device, non_blocking=True)

            images_prompt = images_clean            
            pred_masks = student_model(images_prompt)

            teacher_masks = teacher_logits_batch
            confidence = teacher_scores_batch.clamp(0.0, 1.0)

            loss_teacher = mse_dice_loss(pred_masks, teacher_masks)
            loss_gt = bce_dice_loss(pred_masks, gt_masks)
            loss_area = area_loss(pred_masks, gt_masks)

            
            alpha = confidence.mean()
            loss = alpha * loss_teacher + (1 - alpha) * loss_gt + 0.4 * loss_area

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_iou = compute_iou(pred_masks, gt_masks)

            total_loss += float(loss.item()) * images_clean.size(0)
            total_iou += float(batch_iou) * images_clean.size(0)
            num_samples += images_clean.size(0)

            wandb.log({
                "batch_loss": float(loss.item()),
                "loss_teacher": float(loss_teacher.item()),
                "teacher_confidence_mean": float(confidence.mean().item()),
                "loss_gt": float(loss_gt.item()),
                "batch_mIoU": float(batch_iou),
                "epoch": epoch + 1
            })

            if batch_idx % vis_interval == 0:
                save_visualization(
                    image_clean_tensor=images_clean[0],
                    image_prompt_tensor=images_prompt[0],
                    pred_mask=pred_masks[0:1],
                    teacher_mask=teacher_masks[0:1],
                    prompt_xy=(int(prompts[0, 0].item()), int(prompts[0, 1].item())),
                    step=epoch * len(train_loader) + batch_idx, 
                    IMAGE_OUTPUT_DIR=IMAGE_OUTPUT_DIR
                )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, num_samples),
            "train_mIoU": total_iou / max(1, num_samples),
        })

        
        student_model.eval()
        val_loss, val_iou, val_samples = 0.0, 0.0, 0

        with torch.no_grad():
            for images_clean, gt_masks, prompts, _, teacher_logits_batch, teacher_scores_batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} - Val"):
                images_clean = images_clean.to(device, non_blocking=True)
                gt_masks = gt_masks.to(device, non_blocking=True)
                prompts = prompts.to(device, non_blocking=True)
                teacher_logits_batch = teacher_logits_batch.to(device, non_blocking=True)
                teacher_scores_batch = teacher_scores_batch.to(device, non_blocking=True)

                
                images_prompt = images_clean

                pred_masks = student_model(images_prompt)

                
                teacher_masks = teacher_logits_batch

                alpha = 0.5
                loss_teacher = mse_dice_loss(pred_masks, teacher_masks)
                loss_gt = bce_dice_loss(pred_masks, gt_masks)
                loss = alpha * loss_teacher + (1 - alpha) * loss_gt

                val_loss += float(loss.item()) * images_clean.size(0)
                val_iou += float(compute_iou(pred_masks, gt_masks)) * images_clean.size(0)
                val_samples += images_clean.size(0)

        wandb.log({
            "epoch": epoch + 1,
            "val_loss": val_loss / max(1, val_samples),
            "val_mIoU": val_iou / max(1, val_samples),
        })

        save_path = os.path.join(OUTPUT_DIR, f"{NAME_RUN}_student_epoch{epoch + 1}.pt")
        torch.save(student_model.state_dict(), save_path)

if __name__ == "__main__":
    train()
