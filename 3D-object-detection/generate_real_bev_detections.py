#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

print("="*80)
print("Generating Real PointPillars BEV Detection Visualizations")
print("="*80)

# Download model from MMDetection3D model zoo
print("\n[Step 1] Downloading PointPillars model...")
os.system("mim download mmdet3d --config hv_pointpillars_secfpn_6x8_160e_kitti-3d-car --dest .")

try:
    from mmdet3d.apis import init_model, inference_detector
    print("✓ MMDet3D loaded")
except:
    print("Installing MMDet3D...")
    os.system("pip install mmdet3d -q")
    from mmdet3d.apis import init_model, inference_detector

# Find the downloaded files
config_files = list(Path(".").glob("hv_pointpillars*.py"))
checkpoint_files = list(Path(".").glob("hv_pointpillars*.pth"))

if not config_files or not checkpoint_files:
    print("✗ Model files not found. Using direct download...")
    os.system("wget https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed5ed.pth")
    os.system("wget https://raw.githubusercontent.com/open-mmlab/mmdetection3d/main/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py")
    config_files = list(Path(".").glob("hv_pointpillars*.py"))
    checkpoint_files = list(Path(".").glob("hv_pointpillars*.pth"))

CONFIG_FILE = str(config_files[0]) if config_files else None
CHECKPOINT_FILE = str(checkpoint_files[0]) if checkpoint_files else None

if not CONFIG_FILE or not CHECKPOINT_FILE:
    print("✗ Failed to download model files")
    sys.exit(1)

print(f"✓ Config: {CONFIG_FILE}")
print(f"✓ Checkpoint: {CHECKPOINT_FILE}")

print("\n[Step 2] Loading model...")
try:
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0')
    print("✓ Model loaded on CUDA")
except:
    print("CUDA not available, using CPU...")
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device='cpu')
    print("✓ Model loaded on CPU")

KITTI_DATA = Path("data/kitti/training/velodyne")
NUSCENES_DATA = Path("data/nuscenes_demo/lidar")

KITTI_OUT = Path("outputs/pointpillars_kitti_bev/images")
NUSCENES_OUT = Path("outputs/pointpillars_nuscenes_bev/images")

KITTI_OUT.mkdir(parents=True, exist_ok=True)
NUSCENES_OUT.mkdir(parents=True, exist_ok=True)

def create_bev_viz(points, result, output_path, title):
    """Create BEV visualization"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    sc = ax.scatter(x, y, c=z, s=0.3, cmap='viridis', alpha=0.5)
    
    # Draw boxes
    if hasattr(result, 'pred_instances_3d'):
        pred = result.pred_instances_3d
        if len(pred.scores_3d) > 0:
            boxes = pred.boxes_3d.tensor.cpu().numpy()
            scores = pred.scores_3d.cpu().numpy()
            
            for box, score in zip(boxes, scores):
                if score < 0.3:
                    continue
                
                cx, cy = box[0], box[1]
                dx, dy = box[3], box[4]
                yaw = box[6] if len(box) > 6 else 0
                
                # Box corners
                corners_x = [-dx/2, dx/2, dx/2, -dx/2, -dx/2]
                corners_y = [-dy/2, -dy/2, dy/2, dy/2, -dy/2]
                
                cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
                rx = [cx + (x*cos_yaw - y*sin_yaw) for x, y in zip(corners_x, corners_y)]
                ry = [cy + (x*sin_yaw + y*cos_yaw) for x, y in zip(corners_x, corners_y)]
                
                color = 'red' if score >= 0.7 else ('orange' if score >= 0.5 else 'yellow')
                ax.plot(rx, ry, color=color, linewidth=2)
                ax.text(cx, cy, f'{score:.2f}', color=color, fontsize=9,
                       weight='bold', bbox=dict(facecolor=color, alpha=0.6))
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title, fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.colorbar(sc, label='Height (m)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

print("\n[Step 3] Processing KITTI dataset...")
kitti_files = sorted(list(KITTI_DATA.glob('*.bin')))[:25]
print(f"Found {len(kitti_files)} KITTI samples")

for i, pcd_file in enumerate(kitti_files):
    try:
        print(f"  [{i+1}/{len(kitti_files)}] {pcd_file.name}", end=" ")
        points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
        result = inference_detector(model, str(pcd_file))
        n_det = len(result.pred_instances_3d.scores_3d) if hasattr(result, 'pred_instances_3d') else 0
        
        out_file = KITTI_OUT / f"{pcd_file.stem}_bev.png"
        create_bev_viz(points, result, out_file, f"KITTI - {pcd_file.stem} ({n_det} detections)")
        print(f"- {n_det} objects detected ✓")
    except Exception as e:
        print(f"- Error: {e}")

print("\n[Step 4] Processing nuScenes dataset...")
nuscenes_files = sorted(list(NUSCENES_DATA.glob('*.bin')))[:25]
print(f"Found {len(nuscenes_files)} nuScenes samples")

for i, pcd_file in enumerate(nuscenes_files):
    try:
        print(f"  [{i+1}/{len(nuscenes_files)}] {pcd_file.name}", end=" ")
        points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 5)[:, :4]
        result = inference_detector(model, str(pcd_file))
        n_det = len(result.pred_instances_3d.scores_3d) if hasattr(result, 'pred_instances_3d') else 0
        
        out_file = NUSCENES_OUT / f"{pcd_file.stem}_bev.png"
        create_bev_viz(points, result, out_file, f"nuScenes - {pcd_file.stem} ({n_det} detections)")
        print(f"- {n_det} objects detected ✓")
    except Exception as e:
        print(f"- Error: {e}")

print("\n" + "="*80)
print("COMPLETE! Generated BEV visualizations for all samples")
print("="*80)
print(f"KITTI images: {KITTI_OUT}")
print(f"nuScenes images: {NUSCENES_OUT}")
print(f"Total images: {len(list(KITTI_OUT.glob('*.png'))) + len(list(NUSCENES_OUT.glob('*.png')))}")
