#!/usr/bin/env python3
"""
PointPillars 3D Object Detection with BEV Visualization
Generates Bird's Eye View visualizations similar to the reference image
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import time
import json

try:
    from mmdet3d.apis import init_model, inference_detector
    from mmdet3d.registry import VISUALIZERS
except ImportError:
    print("Error: MMDetection3D not found. Installing...")
    os.system("pip install mmdet3d -q")
    from mmdet3d.apis import init_model, inference_detector

print("="*80)
print("PointPillars 3D Object Detection - BEV Visualization Generator")
print("="*80)

# Configuration
CONFIG_FILE = "checkpoints/kitti_pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed5ed.py"
CHECKPOINT_FILE = "checkpoints/kitti_pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed5ed5e.pth"

KITTI_DATA_DIR = Path("data/kitti/training/velodyne")
NUSCENES_DATA_DIR = Path("data/nuscenes_demo/lidar")

KITTI_OUTPUT_DIR = Path("outputs/pointpillars_kitti_bev")
NUSCENES_OUTPUT_DIR = Path("outputs/pointpillars_nuscenes_bev")

# Create output directories
for dir in [KITTI_OUTPUT_DIR, NUSCENES_OUTPUT_DIR]:
    dir.mkdir(parents=True, exist_ok=True)
    (dir / "images").mkdir(exist_ok=True)
    (dir / "json").mkdir(exist_ok=True)

def create_bev_visualization(points, predictions, output_path, title=""):
    """
    Create Bird's Eye View visualization with detection boxes
    Similar to the reference image provided
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Plot point cloud in BEV (X-Y plane)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Color points by height (Z coordinate)
    scatter = ax.scatter(x, y, c=z, s=0.5, cmap='viridis', alpha=0.6)
    
    # Draw detection boxes
    if 'boxes_3d' in predictions and len(predictions['boxes_3d']) > 0:
        boxes = predictions['boxes_3d'].tensor.cpu().numpy()
        scores = predictions['scores_3d'].cpu().numpy()
        labels = predictions['labels_3d'].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            if score < 0.3:  # Filter low confidence
                continue
            
            # Extract box center and dimensions
            cx, cy, cz = box[0], box[1], box[2]
            dx, dy, dz = box[3], box[4], box[5]
            yaw = box[6] if len(box) > 6 else 0
            
            # Create rotated rectangle for BEV
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            
            # Calculate corners
            corners_x = [-dx/2, dx/2, dx/2, -dx/2, -dx/2]
            corners_y = [-dy/2, -dy/2, dy/2, dy/2, -dy/2]
            
            # Rotate and translate
            rotated_x = [cx + (x*cos_yaw - y*sin_yaw) for x, y in zip(corners_x, corners_y)]
            rotated_y = [cy + (x*sin_yaw + y*cos_yaw) for x, y in zip(corners_x, corners_y)]
            
            # Choose color based on confidence
            if score >= 0.7:
                color = 'red'
                linewidth = 2.5
            elif score >= 0.5:
                color = 'orange'
                linewidth = 2.0
            else:
                color = 'yellow'
                linewidth = 1.5
            
            # Draw box
            ax.plot(rotated_x, rotated_y, color=color, linewidth=linewidth)
            
            # Add score label
            ax.text(cx, cy, f'{score:.2f}', color=color, fontsize=10, 
                   weight='bold', bbox=dict(facecolor=color, alpha=0.7, edgecolor='none'))
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add colorbar for height
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Height (m)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def load_point_cloud(file_path, dataset_type='kitti'):
    """
    Load point cloud from file
    """
    if dataset_type == 'kitti':
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    else:  # nuscenes
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
        points = points[:, :4]  # Use first 4 dimensions
    
    return points

def run_detection_on_dataset(model, data_dir, output_dir, dataset_type='kitti', max_samples=25):
    """
    Run detection on all samples in dataset
    """
    print(f"\n{'='*80}")
    print(f"Processing {dataset_type.upper()} dataset")
    print(f"{'='*80}")
    
    # Get all point cloud files
    pc_files = sorted(list(data_dir.glob('*.bin')))[:max_samples]
    
    print(f"Found {len(pc_files)} samples")
    
    results_summary = []
    
    for idx, pc_file in enumerate(pc_files):
        print(f"\nProcessing [{idx+1}/{len(pc_files)}]: {pc_file.name}")
        
        try:
            # Load point cloud
            points = load_point_cloud(pc_file, dataset_type)
            print(f"  Loaded {len(points)} points")
            
            # Run inference
            start_time = time.time()
            result = inference_detector(model, str(pc_file))
            inference_time = time.time() - start_time
            
            # Extract predictions
            pred_data = result.pred_instances_3d
            num_detections = len(pred_data.scores_3d)
            
            print(f"  Detected {num_detections} objects in {inference_time:.3f}s")
            
            # Create BEV visualization
            sample_name = pc_file.stem
            output_path = output_dir / "images" / f"{sample_name}_bev.png"
            
            title = f"{dataset_type.upper()} - {sample_name} (Detections: {num_detections})"
            create_bev_visualization(points, pred_data, output_path, title=title)
            
            print(f"  ✓ Saved BEV visualization: {output_path}")
            
            # Save JSON predictions
            json_data = {
                'sample_id': sample_name,
                'num_detections': num_detections,
                'inference_time': inference_time,
                'scores': pred_data.scores_3d.cpu().numpy().tolist(),
                'labels': pred_data.labels_3d.cpu().numpy().tolist(),
                'boxes': pred_data.boxes_3d.tensor.cpu().numpy().tolist()
            }
            
            json_path = output_dir / "json" / f"{sample_name}_pred.json"
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            results_summary.append({
                'sample': sample_name,
                'detections': num_detections,
                'time': inference_time
            })
            
        except Exception as e:
            print(f"  ✗ Error processing {pc_file.name}: {e}")
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"{dataset_type.upper()} Processing Complete")
    print(f"{'='*80}")
    print(f"Total samples processed: {len(results_summary)}")
    print(f"Total detections: {sum(r['detections'] for r in results_summary)}")
    print(f"Average inference time: {np.mean([r['time'] for r in results_summary]):.3f}s")
    
    return results_summary

def main():
    print("\n[1/3] Loading PointPillars model...")
    print("-" * 80)
    
    # Initialize model
    try:
        model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0')
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Attempting to use CPU...")
        model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device='cpu')
        print("✓ Model loaded on CPU")
    
    # Process KITTI dataset
    print("\n[2/3] Processing KITTI dataset...")
    print("-" * 80)
    kitti_results = run_detection_on_dataset(
        model, KITTI_DATA_DIR, KITTI_OUTPUT_DIR, 'kitti', max_samples=25
    )
    
    # Process nuScenes dataset
    print("\n[3/3] Processing nuScenes dataset...")
    print("-" * 80)
    nuscenes_results = run_detection_on_dataset(
        model, NUSCENES_DATA_DIR, NUSCENES_OUTPUT_DIR, 'nuscenes', max_samples=25
    )
    
    print("\n" + "="*80)
    print("ALL PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nKITTI BEV images: {KITTI_OUTPUT_DIR}/images/")
    print(f"nuScenes BEV images: {NUSCENES_OUTPUT_DIR}/images/")
    print(f"\nTotal images generated: {len(kitti_results) + len(nuscenes_results)}")
    print("\nYou can now view the BEV visualizations!")

if __name__ == "__main__":
    main()
