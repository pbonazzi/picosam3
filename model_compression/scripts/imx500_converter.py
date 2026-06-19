import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
import numpy as np
import os
import random
from PIL import Image
import model_compression_toolkit as mct
from tqdm import tqdm
import itertools
import onnx

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from picosam2_model_distillation import PicoSAM2
from picosam2_model_distillation import PicoSAM2Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_ckpt = os.path.join(BASE_DIR, "..", "checkpoints", "PicoSAM2_student_epoch5.pt")
onnx_output_path = os.path.join(BASE_DIR, "..", "checkpoints", "PicoSAM2_student_quantized.onnx")
val_images_dir = os.path.join(BASE_DIR, "..", "dataset", "val2017")
annotations_file = os.path.join(BASE_DIR, "..", "dataset", "annotations", "instances_val2017.json")


model = PicoSAM2(in_channels=3)
model.load_state_dict(torch.load(model_ckpt, map_location="cpu"))
model.eval()


dataset = PicoSAM2Dataset(
    image_root=val_images_dir,
    annotation_file=annotations_file,
    image_size=96
)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


def representative_dataset_gen():
    dataloader_iter = itertools.cycle(dataloader)
    for _ in range(10):
        batch = next(dataloader_iter)
        images = batch[0]
        yield [images]


target_platform_cap = mct.get_target_platform_capabilities('pytorch', 'default')

quantized_model, quantization_info = mct.ptq.pytorch_post_training_quantization(
    in_module=model,
    representative_data_gen=representative_dataset_gen,
    target_platform_capabilities=target_platform_cap
)


mct.exporter.pytorch_export_model(
    model=quantized_model,
    save_model_path=onnx_output_path,
    repr_dataset=representative_dataset_gen
)

print(f"Exported quantized ONNX model to {onnx_output_path}")