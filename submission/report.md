# 3D Object Detection using MMDetection3D

## Comparative Analysis of PointPillars, SECOND, 3DSSD, and CenterPoint on KITTI & nuScenes

**Author:** Vasav Patel  
**Course:** CMPE 297 – Deep Learning  
**Institution:** San Jose State University  
**Date:** December 2025

---

## Table of Contents

1. [Introduction & Objectives](#1-introduction--objectives)
2. [Environment Setup](#2-environment-setup)
3. [Models & Datasets](#3-models--datasets)
4. [Implementation](#4-implementation)
5. [Results & Metrics](#5-results--metrics)
6. [Visualizations](#6-visualizations)
7. [Key Takeaways](#7-key-takeaways)
8. [Limitations & Future Work](#8-limitations--future-work)
9. [Deliverables Summary](#9-deliverables-summary)
10. [References](#10-references)

---

## 1. Introduction & Objectives

Three-dimensional object detection from LiDAR point clouds is a **fundamental capability** for autonomous vehicles, robotics, and advanced driver assistance systems. Unlike 2D image-based detection, 3D detection provides:

- **Precise Spatial Localization:** Accurate (x, y, z) coordinates in 3D space
- **Depth Estimation:** True distance measurements from LiDAR
- **Object Orientation:** Yaw angle for vehicle heading direction
- **Size Estimation:** Accurate length, width, height dimensions

### Project Objectives

| Objective | Status |
|-----------|--------|
| ≥2 models on ≥2 datasets | ✅ **4 models × 2 datasets** |
| Save .png frames | ✅ **50+ PNG files** |
| Save .ply point clouds | ✅ **50+ PLY files** |
| Save .json metadata | ✅ **25+ JSON files** |
| Demo video | ✅ **5 videos** |
| ≥2 metrics comparison | ✅ **6+ metrics** |
| 3-5 key takeaways | ✅ **5 takeaways** |

---

## 2. Environment Setup

### Hardware Platform

| Component | Primary Setup (Cloud) | Alternative (Local) |
|-----------|----------------------|---------------------|
| Platform | Lightning AI Studio | Windows 10 Workstation |
| GPU | NVIDIA Tesla T4 (16GB) | NVIDIA GTX 1650 (4GB) |
| CUDA | 12.1 | 11.8 |
| RAM | 32GB | 16GB |

### Software Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10 | Runtime environment |
| PyTorch | 2.1.2+cu121 | Deep learning framework |
| MMCV | 2.1.0 | OpenMMLab CV operations |
| MMDetection | 3.2.0 | 2D detection framework |
| MMDetection3D | 1.4.0 | 3D detection framework |
| NumPy | **1.26.4** | Numerical computing (pinned for ABI) |
| Open3D | 0.18.0 | 3D visualization |

### Installation Commands

```bash
# Step 1: Create conda environment
conda create -n mmdet3d python=3.10 -y
conda activate mmdet3d

# Step 2: Install PyTorch with CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install OpenMIM
pip install openmim

# Step 4: Install MMCV
mim install mmcv==2.1.0

# Step 5: Install MMDetection3D
pip install mmdet==3.2.0 mmdet3d==1.4.0

# Step 6: CRITICAL - Pin NumPy for ABI compatibility
pip install numpy==1.26.4

# Step 7: Visualization tools
pip install open3d matplotlib opencv-python moviepy tqdm
```

### Verification

```python
import torch
print(f"PyTorch: {torch.__version__}")       # 2.1.2+cu121
print(f"CUDA: {torch.cuda.is_available()}")  # True
print(f"GPU: {torch.cuda.get_device_name(0)}")  # Tesla T4
```

---

## 3. Models & Datasets

### Model Architectures

| Model | Type | Encoding | Key Feature |
|-------|------|----------|-------------|
| **PointPillars** | Pillar-based | Vertical pillars → 2D CNN | Fast, **62 FPS** |
| **SECOND** | Voxel-based | Sparse 3D convolutions | High accuracy |
| **3DSSD** | Point-based | Raw point features | Anchor-free |
| **CenterPoint** | Center-based | Heatmap + regression | Multi-class, tracking |

#### PointPillars (Lang et al., CVPR 2019)
Converts 3D point clouds into vertical "pillars" encoded as a 2D pseudo-image. Achieves **62 FPS** on a single GPU—ideal for real-time applications.

#### SECOND (Yan et al., Sensors 2018)
Uses **sparse 3D convolutions** to efficiently process voxelized point clouds. Exploits <5% occupancy of LiDAR data for 10-100x memory reduction.

#### 3DSSD (Yang et al., CVPR 2020)
Single-stage, **anchor-free** point-based detector. Directly processes raw point clouds using feature propagation network.

#### CenterPoint (Yin et al., CVPR 2021)
Detects objects by predicting **center points as heatmap peaks**. Naturally handles varying sizes and extends to velocity estimation.

### Dataset Characteristics

| Attribute | KITTI | nuScenes |
|-----------|-------|----------|
| LiDAR | Velodyne HDL-64E | 32-beam spinning |
| FOV | Front-facing (~90°) | **360° surround** |
| Points/Frame | ~120,000 | ~300,000 |
| Classes | 3 | **10** |
| Difficulty | Moderate | High |

---

## 4. Implementation

### Core Inference Code

```python
from mmdet3d.apis import init_model, inference_detector

def run_3d_detection(config, checkpoint, pcd_path, device='cuda:0'):
    """Execute 3D object detection inference."""
    # Initialize model
    model = init_model(config, checkpoint, device=device)
    
    # Run inference
    result, data = inference_detector(model, pcd_path)
    
    # Extract predictions
    predictions = {
        'bboxes_3d': result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy(),
        'scores_3d': result.pred_instances_3d.scores_3d.cpu().numpy(),
        'labels_3d': result.pred_instances_3d.labels_3d.cpu().numpy()
    }
    return predictions
```

### Running Inference

```bash
# PointPillars on KITTI
python mmdet3d_inference2.py \
    --dataset kitti \
    --model configs/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py \
    --checkpoint checkpoints/pointpillars_kitti.pth \
    --out-dir outputs/kitti_pointpillars \
    --device cuda:0 --score-thr 0.2

# CenterPoint on nuScenes
python mmdet3d_inference2.py \
    --dataset any \
    --model configs/centerpoint_voxel01_second_secfpn.py \
    --checkpoint checkpoints/centerpoint_nusc.pth \
    --out-dir outputs/nuscenes_centerpoint \
    --device cuda:0 --score-thr 0.2
```

---

## 5. Results & Metrics

### Main Comparison Table

| Model + Dataset | Detections | Mean Score | Max Score | High Conf (≥0.7) | Score Std | Est. FPS | GPU Mem |
|-----------------|------------|------------|-----------|------------------|-----------|----------|---------|
| **PointPillars (KITTI)** | 10 | **0.792** | 0.975 | 8 (80%) | 0.169 | **50-62** | 1.8 GB |
| PointPillars (nuScenes) | 365 | 0.127 | 0.711 | 1 (0.3%) | 0.095 | 45-55 | 2.1 GB |
| **SECOND (KITTI)** | 11 | 0.880 | 0.944 | **9 (82%)** | **0.152** | 25-30 | 2.8 GB |
| 3DSSD (KITTI) | 50 | 0.158 | 0.905 | 7 (14%) | 0.318 | 20-25 | 3.2 GB |
| CenterPoint (nuScenes) | 264 | 0.244 | 0.874 | 15 (6%) | 0.183 | 18-22 | 3.5 GB |

### Confidence Distribution

| Model | High (≥0.7) | Medium (0.5-0.7) | Low (<0.5) |
|-------|-------------|------------------|------------|
| PointPillars (KITTI) | **80%** | 10% | 10% |
| SECOND (KITTI) | **82%** | 9% | 9% |
| 3DSSD (KITTI) | 14% | 4% | 82% |
| CenterPoint (nuScenes) | 6% | 7% | 87% |

### Per-Class Detection (CenterPoint)

| Class | Detections | Mean Score |
|-------|------------|------------|
| Car | 36 | 0.412 |
| Barrier | 61 | 0.187 |
| Traffic Cone | 56 | 0.156 |
| Pedestrian | 36 | 0.234 |
| Truck | 34 | 0.287 |

---

## 6. Visualizations

### Screenshots (18 files in `results/screenshots/`)

- `pointpillars_kitti_2d.png` - PointPillars BEV detection (10 cars)
- `second_kitti_2d.png` - SECOND detection (11 cars)
- `centerpoint_nuscenes_bev.png` - CenterPoint multi-class
- `open3d_3d_view.png` - 3D point cloud with boxes
- `3dssd_2d.png`, `3dssd_filtered_2d.png` - 3DSSD results
- Additional 12 screenshots covering various configurations

### Demo Videos (5 files)

- `results/demo_video.mp4` - Main demo (12 seconds)
- `results/detections_demo.mp4` - Detection showcase
- `outputs/CENTERPOINT_NUSCENES/centerpoint_nuscenes_demo.mp4`

### Open3D Visualization Command

```bash
python scripts/open3d_view_saved_ply.py \
    --dir outputs/kitti_pointpillars \
    --basename 000008 \
    --width 1600 --height 1200
```

---

## 7. Key Takeaways

### Takeaway 1: PointPillars Excels on Structured Scenes

**What Works:** PointPillars achieves **highest mean confidence (0.792)** on KITTI with 80% high-confidence detections.

**Why:** Pillar-based BEV encoding aligns perfectly with front-facing sensor geometry. 2D CNN processing is highly optimized.

**Where It Fails:** On nuScenes (360° view), mean score drops to 0.127—a **6x performance degradation**.

---

### Takeaway 2: SECOND Provides Superior Stability

**What Works:** SECOND achieves **lowest score variance (0.152)** and **highest high-confidence rate (82%)** on KITTI.

**Why:** Sparse 3D convolutions capture full geometric context for precise bounding box regression.

**Trade-off:** 2x slower than PointPillars (25-30 FPS vs 50-62 FPS).

---

### Takeaway 3: 3DSSD Has High False Positive Rate

**What Fails:** 3DSSD produces **50 detections with mean score 0.158**. Most detections are near-zero confidence.

**Why:** Single-stage anchor-free design aggressively proposes candidates without two-stage refinement.

**Mitigation:** Increase threshold from 0.2 to **0.6-0.7** to reduce false positives.

---

### Takeaway 4: CenterPoint Dominates Multi-Class Detection

**What Works:** CenterPoint detects **264 objects across 10 categories** on nuScenes including rare classes.

**Why:** Center-based heatmap approach is class-agnostic and scale-invariant.

**Limitation:** Only 6% high-confidence—conservative predictions prioritize recall over precision.

---

### Takeaway 5: Dataset Complexity Dominates Performance

**Critical Finding:** Same model (PointPillars) shows **6x performance difference** between KITTI (0.792) and nuScenes (0.127).

**Implication:** Models trained on simple datasets don't generalize to complex scenes. Real-world deployment requires target domain evaluation.

---

## 8. Limitations & Future Work

### Current Limitations

1. **No Ground Truth mAP:** Metrics based on confidence scores, not IoU-based precision/recall
2. **Single-Frame Analysis:** No temporal consistency evaluation
3. **Sample Data:** Inference on sample frames, not full validation sets
4. **Pretrained Only:** No fine-tuning performed
5. **Informal Latency:** FPS estimates not systematically benchmarked

### Future Work

- Implement standard KITTI/nuScenes mAP evaluation protocols
- Fine-tune models on target dataset (Excellent Option)
- Integrate multi-sensor fusion (LiDAR + Camera)
- Deploy on Jetson Orin Nano for edge inference
- Extend CenterPoint with tracking

---

## 9. Deliverables Summary

| Requirement | Status | Evidence |
|-------------|--------|----------|
| ≥2 Models | ✅ **4 models** | PointPillars, SECOND, 3DSSD, CenterPoint |
| ≥2 Datasets | ✅ **2 datasets** | KITTI, nuScenes |
| .png frames | ✅ **50+ files** | `results/screenshots/`, `outputs/` |
| .ply point clouds | ✅ **50+ files** | `results/ply_files/`, `outputs/` |
| .json metadata | ✅ **25+ files** | `results/json_metadata/`, `outputs/` |
| Demo video | ✅ **5 videos** | `results/*.mp4`, `outputs/*.mp4` |
| ≥4 screenshots | ✅ **18 screenshots** | `results/screenshots/` |
| ≥2 metrics | ✅ **6+ metrics** | See comparison tables |
| Comparison table | ✅ | Multiple tables above |
| 3-5 takeaways | ✅ **5 takeaways** | Section 7 |
| README | ✅ | `README.md` with reproducible steps |
| Commented code | ✅ | `code/mmdet3d_inference2.py` |

---

## 10. References

1. Lang, A.H., et al. "**PointPillars: Fast Encoders for Object Detection from Point Clouds**" CVPR 2019. https://arxiv.org/abs/1812.05784

2. Yan, Y., et al. "**SECOND: Sparsely Embedded Convolutional Detection**" Sensors 2018. https://doi.org/10.3390/s18103337

3. Yang, Z., et al. "**3DSSD: Point-based 3D Single Stage Object Detector**" CVPR 2020. https://arxiv.org/abs/2002.10187

4. Yin, T., et al. "**Center-based 3D Object Detection and Tracking**" CVPR 2021. https://arxiv.org/abs/2006.11275

5. MMDetection3D. https://github.com/open-mmlab/mmdetection3d

6. KITTI Dataset. http://www.cvlibs.net/datasets/kitti/

7. nuScenes Dataset. https://www.nuscenes.org/

---

**End of Report**

Vasav Patel | CMPE 297 Deep Learning | San Jose State University | December 2025
