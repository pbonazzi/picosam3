# In-Sensor Centered Image Segmentation

### [📜 PicoSAM2](https://arxiv.org/pdf/2506.18807)  | 📜 [PicoSAM3](https://arxiv.org/pdf/2603.11917)

**PicoSAM2** and **PicoSAM3** are minimal, segmentation model distilled from Meta’s SAM 2.1 and SAM3 — purpose-built for deployment on edge devices such as the **Sony IMX500**. It replicates the implicit centered image segmentation while drastically reducing model size and computational cost, making real-time inference feasible on low-power hardware.

>  ~1.2MB quantized model  
>  Real-time inference on embedded devices  
>  Supports implicit point-based prompt segmentation  
>  Distilled from SAM2.1  Hiera Tiny and SAM3

---
## References & Citation ✉️ 

This repository reproduces the results from our article, which received the Outstanding Lecture Award at IEEE Sensors 2025 in Vancouver.
If you find this work useful please cite us with the following:
``` 
@article{picosam3_2026,
      title={PicoSAM3: Real-Time In-Sensor Region-of-Interest Segmentation}, 
      author={Pietro Bonazzi and Nicola Farronato and Stefan Zihlmann and Haotong Qin and Michele Magno},
      journal={IEEE Sensors Journal},
      year={2026}
}

@article{picosam2_2025,
      title={PicoSAM2: Low-Latency Segmentation In-Sensor for Edge Vision Applications}, 
      author={Bonazzi, Pietro and Farronato, Nicola and Zihlmann, Stefan and Qin, Haotong and Magno, Michele},
      journal={IEEE Sensors Conference}, 
      year={2025}
}
```

Leave a star to support our open source initiative!⭐️ 

## Quick Start

PicoSAM2 and PicoSAM3 come with an automated setup script `init.sh` to get everything ready in one step. It:

- Downloads **COCO 2017** validation and training images  
- Downloads **LVIS v1** validation annotations  
- Unzips and organizes all files under a structured `dataset/` folder  
- Downloads all **SAM 2.1** model checkpoints into `checkpoints/`   
- Installs all required dependencies from `requirements.txt`  
- Installs the project into the environment  


### To get started, simply run:

```bash
./init.sh
```
You do **not** need to manually download any datasets or checkpoints — everything is handled by the script.

Afterwards (optional if you just want to use the compression pipeline) download the pretrained PicoSAM2 weights from this [Zenodo folder](https://zenodo.org/records/15728470) and add them to the checkpoints directory.

## How to Use

Once the setup is complete, activate the environment:

```bash
source .venv/bin/activate
```

Then you can run any script inside the `model_compression/` folder:

```bash 
python3 -m model_compression.scripts.picosam3_model_distillation       # Distill student from SAM3
python3 -m model_compression.scripts.picosam3_train_from_scratch       # Train supervised baseline
python3 -m model_compression.scripts.benchmark                         # Evaluate mIoU, mAP 
python3 -m model_compression.scripts.imx500_converter                  # Export ONNX for IMX500
```

## Deployment on the IMX500

From now on, follow the official deployment documentation of the raspberrypi AI Camera: https://www.raspberrypi.com/documentation/accessories/ai-camera.html#model-deployment

The model converted to IMX500 format from the onnx (Can be done still on your computer): 
```bash
imxconv-pt -i picosam2_student_quantized.onnx -o . --overwrite-output
```
 Result is a PackerOut.zip which then should be loaded onto the raspberry pi.

 Afterwards, on the raspberrypi, run:
 ```bash
imx500-package -i packerOut.zip -o .
```
This creates a network.rpk file, which afterwards can be deployed on the IMX500 using the Picamera2 Script.

## Pretrained Checkpoints (Example PicoSam2)

After setup, the following files are available under `checkpoints/`:

| File                             | Description                            |
|----------------------------------|----------------------------------------|
| `PicoSAM2_student_epoch1.pt`     | Student model trained via distillation (Google Drive) |
| `PicoSAM2_student_quantized.onnx`| Quantized export ready for IMX500 conversion (Google Drive) |
| `PicoSAM2_epoch1.pt`             | Supervised baseline (Google Drive)                    |
| `sam2.1_hiera_*.pt`              | SAM 2.1 teacher models (Tiny → Large) |

These are ready for use in training, benchmarking, or deployment.

## Qualitative Results

Side-by-side segmentation comparisons between PicoSAM3 and baseline models on COCO images:

| Scene | Comparison |
|-------|------------|
| Airplane | [comparison_airplane.pdf](model_compression/qualitative_results/comparison_airplane.pdf) |
| Apple | [comparison_apple.pdf](model_compression/qualitative_results/comparison_apple.pdf) |
| Bench | [comparison_bench.pdf](model_compression/qualitative_results/comparison_bench.pdf) |
| Zebra | [comparison_zebra.pdf](model_compression/qualitative_results/comparison_zebra.pdf) |

## Directory Structure

```
.
├── checkpoints/                     # Pretrained models & teacher weights
│   ├── PicoSAM2_epoch1.pt
│   ├── PicoSAM2_student_epoch1.pt
│   ├── PicoSAM2_student_quantized.onnx
│   ├── sam2.1_hiera_tiny.pt
│   ├── sam2.1_hiera_small.pt
│   ├── sam2.1_hiera_base_plus.pt
│   ├── sam2.1_hiera_large.pt
│   └── download_ckpts.sh
│
├── dataset/                         # COCO + LVIS data
│   ├── train2017/
│   ├── val2017/
│   ├── val2017_lvis/
│   └── annotations/
│       └── lvis_v1_val.json
│
├── model_compression/               # All model code
│   ├── benchmark.py
│   ├── imx500_converter.py
│   ├── picosam2_model_distillation.py
│   ├── picosam2_train_from_scratch.py
│   ├── plot_latency_vs_size.py
│   ├── plot_map_vs_size.py
│   ├── plot_miou_vs_size.py
│   ├── plot_size_comparison.py
│   └── requirements.txt
│
├── init.sh                          # Setup script (data, env, install)
```
