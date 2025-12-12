# 3D Object Detection using MMDetection3D

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1](https://img.shields.io/badge/pytorch-2.1.2-red.svg)](https://pytorch.org/)

> **3D Object Detection** using PointPillars, SECOND, 3DSSD, and CenterPoint on KITTI & nuScenes datasets

**Author:** Vasav Patel  
**Course:** CMPE 249 - Intelligent Autonomous Systems
**Institution:** San Jose State University  
**Date:** December 2025

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Models & Datasets](#models--datasets)
- [Quick Start](#quick-start)
- [Running Inference](#running-inference)
- [Results](#results)
- [Visualizations](#visualizations)
- [Key Takeaways](#key-takeaways)
- [Deliverables](#deliverables)
- [References](#references)

---

## 🎯 Overview

This project demonstrates end-to-end **3D object detection** for autonomous driving using four state-of-the-art deep learning models:

| Model | Type | Key Feature |
|-------|------|-------------|
| **PointPillars** | Pillar-based | Fast, real-time capable (62 FPS) |
| **SECOND** | Voxel-based | High accuracy, sparse 3D convolutions |
| **3DSSD** | Point-based | Single-stage, anchor-free |
| **CenterPoint** | Center-based | Best for tracking, velocity estimation |

**Datasets Used:**
- **KITTI** - Front-facing LiDAR, highway/suburban scenes, 3 classes
- **nuScenes** - 360° LiDAR, dense urban traffic, 10 classes

---

## 📁 Project Structure

```
3DObjectDetection/
├── submission/                    # 📦 Ready-to-submit folder
│   ├── README.md                  # Setup instructions
│   ├── report.md                  # Markdown report
│   ├── code/                      # Inference scripts
│   │   ├── mmdet3d_inference2.py  # Main inference script
│   │   ├── compare_models_metrics.py
│   │   └── scripts/
│   ├── results/                   # All deliverables
│   │   ├── demo_video.mp4         # Demo videos
│   │   ├── screenshots/           # 18 labeled screenshots
│   │   ├── ply_files/             # Point cloud samples
│   │   └── json_metadata/         # Prediction JSONs
│   └── outputs/                   # Full inference outputs
│
├── 3D-object-detection/           # Main inference workspace
│   ├── mmdet3d_inference2.py      # Core inference script
│   ├── compare_models_metrics.py  # Model comparison script
│   ├── outputs/                   # Inference results
│   │   ├── kitti_pointpillars/    # PointPillars on KITTI
│   │   ├── nuscenes_pointpillars/ # PointPillars on nuScenes
│   │   ├── 3dssd/                 # 3DSSD on KITTI
│   │   └── nuscenes_centerpoint/  # CenterPoint on nuScenes
│   ├── scripts/
│   │   └── open3d_view_saved_ply.py  # Open3D visualization
│   └── results/                   # Demo videos & screenshots
│
├── 3d_detection_workspace/        # Secondary workspace
│   ├── scripts/                   # Additional inference scripts
│   ├── results/                   # Visualizations & metadata
│   └── configs/                   # Model configurations
│
├── assignment_code/               # CenterPoint outputs
│   ├── CENTERPOINT_KITTY/         # CenterPoint on KITTI
│   └── CENTERPOINT_NUSCENES/      # CenterPoint on nuScenes
│
├── images/                        # Report images
│   ├── pointpillars_kitti_2d.png
│   ├── second_kitti_2d.png
│   ├── centerpoint_nuscenes_bev.png
│   └── open3d_3d_view.png
│
└── README.md                      # This file
```

---

## 💻 Requirements

### Hardware
- **GPU:** NVIDIA GPU with CUDA support (Tesla T4, GTX 1650+)
- **RAM:** 16GB minimum
- **Storage:** 10GB free space

### Software
| Package | Version |
|---------|---------|
| Python | 3.10 |
| PyTorch | 2.1.2 |
| CUDA | 11.8 or 12.1 |
| MMCV | 2.1.0 |
| MMDetection3D | 1.4.0 |
| NumPy | 1.26.4 |
| Open3D | 0.18.0 |

---

## 🛠️ Installation

### Step 1: Create Conda Environment

```bash
# Create new environment
conda create -n mmdet3d python=3.10 -y
conda activate mmdet3d
```

### Step 2: Install PyTorch with CUDA

```bash
# For CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# OR for CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install MMDetection3D

```bash
# Install OpenMIM
pip install openmim

# Install MMCV
mim install mmcv==2.1.0

# Install MMDetection and MMDetection3D
pip install mmdet==3.2.0 mmdet3d==1.4.0
```

### Step 4: Pin NumPy (Critical!)

```bash
# IMPORTANT: Pin NumPy for ABI compatibility
pip install numpy==1.26.4
```

### Step 5: Install Visualization Tools

```bash
pip install open3d matplotlib opencv-python moviepy tqdm
```

### Step 6: Verify Installation

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

**Expected Output:**
```
PyTorch: 2.1.2+cu121
CUDA available: True
GPU: NVIDIA Tesla T4
```

---

## 🤖 Models & Datasets

### Model Architectures

| Model | Type | Encoding | Key Innovation |
|-------|------|----------|----------------|
| **PointPillars** | Pillar-based | Vertical pillars → 2D CNN | Fast BEV encoding, **62 FPS** |
| **SECOND** | Voxel-based | Sparse 3D convolutions | High accuracy, geometric detail |
| **3DSSD** | Point-based | Raw point features | Anchor-free, single-stage |
| **CenterPoint** | Center-based | Heatmap + regression | Multi-class, velocity estimation |

### Model Configurations

| Model | Dataset | Checkpoint | Config |
|-------|---------|------------|--------|
| PointPillars | KITTI | `pointpillars_kitti.pth` | `pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py` |
| PointPillars | nuScenes | `pointpillars_nus.pth` | `pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py` |
| SECOND | KITTI | `second_kitti_car.pth` | `second_hv_secfpn_8xb6-80e_kitti-3d-car.py` |
| 3DSSD | KITTI | `3dssd_kitti.pth` | `3dssd_4x4_kitti-3d-car.py` |
| CenterPoint | nuScenes | `centerpoint_nusc.pth` | `centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py` |

### Download Checkpoints

```bash
# Using OpenMIM
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest checkpoints/
mim download mmdet3d --config second_hv_secfpn_8xb6-80e_kitti-3d-car --dest checkpoints/
mim download mmdet3d --config 3dssd_4x4_kitti-3d-car --dest checkpoints/
mim download mmdet3d --config centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d --dest checkpoints/
```

---

## 🚀 Quick Start

### Run PointPillars on KITTI (Fastest)

```bash
cd 3D-object-detection

python mmdet3d_inference2.py \
  --dataset kitti \
  --input-path data/kitti/training \
  --frame-number 000008 \
  --model checkpoints/kitti_pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py \
  --checkpoint checkpoints/kitti_pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.pth \
  --out-dir outputs/kitti_pointpillars \
  --device cuda:0 \
  --headless \
  --score-thr 0.2
```

---

## 🔄 Running Inference

### PointPillars on KITTI

```bash
python mmdet3d_inference2.py \
  --dataset kitti \
  --input-path data/kitti/training \
  --frame-number 000008 \
  --model checkpoints/kitti_pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py \
  --checkpoint checkpoints/kitti_pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.pth \
  --out-dir outputs/kitti_pointpillars_gpu \
  --device cuda:0 \
  --headless \
  --score-thr 0.2
```

### CenterPoint on nuScenes

```bash
python mmdet3d_inference2.py \
  --dataset any \
  --input-path data/nuscenes_demo/lidar/sample.pcd.bin \
  --model checkpoints/nuscenes_centerpoint/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  --checkpoint checkpoints/nuscenes_centerpoint/centerpoint_nusc.pth \
  --out-dir outputs/nuscenes_centerpoint \
  --device cuda:0 \
  --headless \
  --score-thr 0.2
```

### SECOND on KITTI

```bash
python mmdet3d_inference2.py \
  --dataset kitti \
  --input-path data/kitti/training \
  --frame-number 000008 \
  --model checkpoints/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py \
  --checkpoint checkpoints/second/second_kitti_car.pth \
  --out-dir outputs/second_kitti \
  --device cuda:0 \
  --headless \
  --score-thr 0.2
```

### 3DSSD on KITTI

```bash
python mmdet3d_inference2.py \
  --dataset kitti \
  --input-path data/kitti/training \
  --frame-number 000008 \
  --model checkpoints/3dssd/3dssd_4x4_kitti-3d-car.py \
  --checkpoint checkpoints/3dssd/3dssd_kitti.pth \
  --out-dir outputs/3dssd \
  --device cuda:0 \
  --headless \
  --score-thr 0.6
```

### View Results with Open3D

```bash
python scripts/open3d_view_saved_ply.py \
  --dir outputs/kitti_pointpillars_gpu \
  --basename 000008 \
  --width 1600 \
  --height 1200
```

---

## 📊 Results

### Performance Comparison Table

| Model + Dataset | Detections | Mean Score | Max Score | High Conf (≥0.7) | Score Std | Est. FPS |
|-----------------|------------|------------|-----------|------------------|-----------|----------|
| **PointPillars (KITTI)** | 10 | **0.792** | 0.975 | 8 (80%) | 0.169 | 50-62 |
| PointPillars (nuScenes) | 365 | 0.127 | 0.711 | 1 (0.3%) | 0.095 | 45-55 |
| **SECOND (KITTI)** | 11 | 0.880 | 0.944 | **9 (82%)** | **0.152** | 25-30 |
| 3DSSD (KITTI) | 50 | 0.158 | 0.905 | 7 (14%) | 0.318 | 20-25 |
| CenterPoint (nuScenes) | 264 | 0.244 | 0.874 | 15 (6%) | 0.183 | 18-22 |

### Best Performers

- **Highest Mean Score:** PointPillars on KITTI (0.792)
- **Most Accurate:** SECOND on KITTI (82% high-confidence)
- **Best Multi-Class:** CenterPoint on nuScenes (264 detections, 10 classes)
- **Fastest:** PointPillars (50-62 FPS)

---

## 🖼️ Visualizations

### Output Files Generated

Each inference run produces:

| File | Description |
|------|-------------|
| `*_predictions.json` | Raw predictions (bboxes, scores, labels) |
| `*_2d_vis.png` | 2D BEV visualization |
| `*_points.ply` | Point cloud (Open3D format) |
| `*_pred_bboxes.ply` | 3D bounding boxes |
| `*_open3d.png` | 3D visualization screenshot |

### Demo Videos

- `3D-object-detection/outputs/detections_demo.mp4`
- `3D-object-detection/results/demo_video.mp4`
- `assignment_code/CENTERPOINT_NUSCENES/centerpoint_nuscenes_demo.mp4`
- `submission/results/demo_video.mp4`

### Screenshots (18+ total)

```
submission/results/screenshots/
├── pointpillars_kitti_2d.png
├── second_kitti_2d.png
├── centerpoint_nuscenes_bev.png
├── open3d_3d_view.png
├── 3dssd_2d.png
├── kitti_pointpillars_3d.png
└── ... (12 more)
```

---

## 💡 Key Takeaways

### 1. PointPillars Excels on KITTI
- Achieves **highest mean confidence (0.792)** with 80% high-confidence detections
- Pillar-based BEV encoding aligns perfectly with front-facing sensor geometry
- **Best for:** Real-time robotics, embedded systems

### 2. SECOND Provides Superior Accuracy
- Sparse 3D convolutions enable detailed geometric reasoning
- **82% high-confidence detections** - most accurate on KITTI
- **Best for:** Offline analysis, precision-critical applications

### 3. 3DSSD Has High False Positive Rate
- Produces 50 detections with mean score of only 0.158
- **Mitigation:** Use score threshold ≥0.6 to reduce false positives
- **Best for:** Research, not production deployment

### 4. CenterPoint Dominates nuScenes
- Center-based detection handles 360° multi-class scenes effectively
- Detected **264 objects across 10 categories**
- **Best for:** Urban autonomous driving, tracking applications

### 5. Dataset Complexity Impacts Performance
- PointPillars drops from 0.792 (KITTI) to 0.127 (nuScenes) - **6x degradation**
- Simpler architectures struggle with dense urban environments
- **Implication:** Models don't generalize across datasets

---

## ✅ Deliverables

| Requirement | Status | Location |
|-------------|--------|----------|
| ≥2 Models | ✅ **4 models** | PointPillars, SECOND, 3DSSD, CenterPoint |
| ≥2 Datasets | ✅ **2 datasets** | KITTI, nuScenes |
| .png frames | ✅ **50+ files** | `outputs/*/000008_2d_vis.png` |
| .ply point clouds | ✅ **50+ files** | `outputs/*/*_points.ply` |
| .json metadata | ✅ **25+ files** | `outputs/*/*_predictions.json` |
| Demo video | ✅ **5 videos** | `outputs/detections_demo.mp4` |
| ≥4 screenshots | ✅ **18 screenshots** | `submission/results/screenshots/` |
| Comparison table | ✅ | See Results section |
| 3-5 takeaways | ✅ **5 takeaways** | See Key Takeaways |
| README | ✅ | This file |
| report.md | ✅ | `submission/report.md` |

---

## 🔧 Troubleshooting

### MMCV Import Error
```bash
# Solution: Reinstall MMCV
pip uninstall -y mmcv
mim install mmcv==2.1.0
```

### NumPy Version Conflict
```bash
# Solution: Pin NumPy
pip install numpy==1.26.4
```

### CUDA Not Available
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with correct CUDA version
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

### Model Not Found
```bash
# Download using MIM
mim download mmdet3d --config <config_name> --dest checkpoints/
```

### 3DSSD False Positives
```bash
# Use higher score threshold
--score-thr 0.6  # or 0.7
```

---

## 📚 References

1. **PointPillars:** Lang, A.H. et al. "PointPillars: Fast Encoders for Object Detection from Point Clouds." CVPR 2019. [arXiv](https://arxiv.org/abs/1812.05784)

2. **SECOND:** Yan, Y. et al. "SECOND: Sparsely Embedded Convolutional Detection." Sensors 2018. [DOI](https://doi.org/10.3390/s18103337)

3. **3DSSD:** Yang, Z. et al. "3DSSD: Point-based 3D Single Stage Object Detector." CVPR 2020. [arXiv](https://arxiv.org/abs/2002.10187)

4. **CenterPoint:** Yin, T. et al. "Center-based 3D Object Detection and Tracking." CVPR 2021. [arXiv](https://arxiv.org/abs/2006.11275)

5. **MMDetection3D:** https://github.com/open-mmlab/mmdetection3d

6. **KITTI Dataset:** http://www.cvlibs.net/datasets/kitti/

7. **nuScenes Dataset:** https://www.nuscenes.org/

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Vasav Patel**  
- GitHub: [@Vasu2604](https://github.com/Vasu2604)
- Course: CMPE 249 - Intelligent Autonomous Systems
- Institution: San Jose State University

---

<p align="center">
  <b>⭐ Star this repo if you found it helpful!</b>
</p>
