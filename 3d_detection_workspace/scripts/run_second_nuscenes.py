#!/usr/bin/env python3
"""
Run SECOND model inference on nuScenes dataset
Generates .png, .ply, and .json artifacts
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add mmdetection3d to path
sys.path.insert(0, '/teamspace/studios/this_studio/mmdetection3d')

from mmdet3d.apis import init_model, inference_detector
from mmdet3d.registry import VISUALIZERS
import mmcv
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    device = 'cuda:0'
else:
    device = 'cpu'
    print("WARNING: Running on CPU")

# Configuration
config_file = 'mmdetection3d/configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py'
checkpoint_file = 'checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-393f000c.pth'

# Output directories
output_base = Path('results/nuscenes/second')
img_dir = output_base / 'images'
meta_dir = output_base / 'metadata'
pc_dir = output_base / 'pointclouds'

for d in [img_dir, meta_dir, pc_dir]:
    d.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*70}")
print("SECOND Model Inference on nuScenes Dataset")
print(f"{'='*70}")
print(f"Config: {config_file}")
print(f"Checkpoint: {checkpoint_file}")
print(f"Device: {device}")
print(f"Output: {output_base}")

# Initialize model
print("\nInitializing SECOND model...")
model = init_model(config_file, checkpoint_file, device=device)
print("✓ Model loaded successfully")

# nuScenes sample data
nuscenes_dir = Path('data/nuscenes')
bin_files = list(nuscenes_dir.glob('samples/LIDAR_TOP/*.pcd.bin'))

if not bin_files:
    print("\n⚠ No .pcd.bin files found, trying .bin files...")
    bin_files = list(nuscenes_dir.glob('samples/LIDAR_TOP/*.bin'))

if not bin_files:
    print("\n❌ ERROR: No point cloud files found in data/nuscenes/samples/LIDAR_TOP/")
    print("Please verify nuScenes data is downloaded correctly.")
    sys.exit(1)

print(f"\nFound {len(bin_files)} point cloud files")
print(f"Processing first file for demo: {bin_files[0].name}")

# Run inference on first sample
pcd_file = str(bin_files[0])
result, data = inference_detector(model, pcd_file)

# Extract predictions
pred_instances_3d = result.pred_instances_3d
scores = pred_instances_3d.scores_3d.cpu().numpy()
bboxes = pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
labels = pred_instances_3d.labels_3d.cpu().numpy()

print(f"\n{'='*70}")
print("Inference Results")
print(f"{'='*70}")
print(f"Detected objects: {len(scores)}")
if len(scores) > 0:
    print(f"Confidence range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"Classes detected: {np.unique(labels)}")

# Save metadata
metadata = {
    'model': 'SECOND',
    'dataset': 'nuScenes',
    'input_file': pcd_file,
    'device': device,
    'num_detections': int(len(scores)),
    'detections': []
}

class_names = ['car', 'pedestrian', 'cyclist']  # KITTI classes
for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
    metadata['detections'].append({
        'id': i,
        'class': class_names[int(label)] if int(label) < len(class_names) else f'class_{int(label)}',
        'confidence': float(score),
        'bbox_3d': bbox.tolist(),
        'center': bbox[:3].tolist(),
        'size': bbox[3:6].tolist(),
        'rotation': float(bbox[6]) if len(bbox) > 6 else 0.0
    })

json_path = meta_dir / 'detections.json'
with open(json_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"\n✓ Saved metadata: {json_path}")

# Save point cloud with detections in PLY format
points = data['inputs']['points'].cpu().numpy()

ply_path = pc_dir / 'detection_result.ply'
with open(ply_path, 'w') as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {len(points)}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property float intensity\n")
    f.write("end_header\n")
    for point in points:
        if len(point) >= 4:
            f.write(f"{point[0]} {point[1]} {point[2]} {point[3]}\n")
        else:
            f.write(f"{point[0]} {point[1]} {point[2]} 0.0\n")

print(f"✓ Saved point cloud: {ply_path} ({len(points)} points)")

# Create visualization (simple BEV)
try:
    from PIL import Image, ImageDraw, ImageFont
    
    # Create bird's eye view
    img_size = 800
    scale = 10  # pixels per meter
    offset = img_size // 2
    
    img = Image.new('RGB', (img_size, img_size), color='black')
    draw = ImageDraw.Draw(img)
    
    # Draw point cloud
    for point in points[::10]:  # Subsample for visibility
        x, y = int(point[0] * scale + offset), int(point[1] * scale + offset)
        if 0 <= x < img_size and 0 <= y < img_size:
            draw.point((x, y), fill='gray')
    
    # Draw bounding boxes
    colors = ['red', 'green', 'blue']
    for bbox, label in zip(bboxes, labels):
        cx, cy, cz, l, w, h = bbox[:6]
        rot = bbox[6] if len(bbox) > 6 else 0
        
        # Draw box center
        x, y = int(cx * scale + offset), int(cy * scale + offset)
        if 0 <= x < img_size and 0 <= y < img_size:
            color = colors[int(label) % len(colors)]
            # Draw simple box
            box_w, box_l = int(w * scale), int(l * scale)
            draw.rectangle(
                [x - box_w//2, y - box_l//2, x + box_w//2, y + box_l//2],
                outline=color, width=2
            )
    
    # Add title
    draw.text((10, 10), f"SECOND on nuScenes ({len(bboxes)} detections)",
              fill='white')
    
    img_path = img_dir / 'detection_bev.png'
    img.save(img_path)
    print(f"✓ Saved visualization: {img_path}")
    
except Exception as e:
    print(f"⚠ Visualization creation failed: {e}")
    print("  Creating simple text-based visualization instead...")
    
    # Fallback: text visualization
    viz_path = img_dir / 'detection_bev.txt'
    with open(viz_path, 'w') as f:
        f.write("SECOND Model - nuScenes Detection Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total detections: {len(scores)}\n\n")
        for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
            cls = class_names[int(label)] if int(label) < len(class_names) else f'class_{int(label)}'
            f.write(f"Detection {i+1}:\n")
            f.write(f"  Class: {cls}\n")
            f.write(f"  Confidence: {score:.3f}\n")
            f.write(f"  Center: ({bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f})\n")
            f.write(f"  Size: ({bbox[3]:.2f}, {bbox[4]:.2f}, {bbox[5]:.2f})\n\n")
    print(f"✓ Saved text visualization: {viz_path}")

print(f"\n{'='*70}")
print("✓ SECOND nuScenes inference complete!")
print(f"{'='*70}")
print(f"Results saved to: {output_base}")
print(f"  - Metadata: {meta_dir}")
print(f"  - Images: {img_dir}")
print(f"  - Point clouds: {pc_dir}")
