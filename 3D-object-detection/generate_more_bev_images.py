import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
import glob
from pathlib import Path

def load_kitti_points(bin_file):
    """Load KITTI point cloud"""
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]

def load_predictions(json_path):
    """Load prediction JSON"""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_bev_plot(points, predictions, title, output_path):
    """Create BEV visualization with bounding boxes"""
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot point cloud
    ax.scatter(points[:, 0], points[:, 1], s=0.5, c=points[:, 2], 
               cmap='viridis', alpha=0.3, vmin=-3, vmax=2)
    
    # Draw detected bounding boxes
    if 'boxes_3d' in predictions and 'scores_3d' in predictions:
        boxes = np.array(predictions['boxes_3d'])
        scores = np.array(predictions['scores_3d'])
        
        for box, score in zip(boxes, scores):
            if len(box) >= 7:
                x, y, z, l, w, h, yaw = box[:7]
                
                # Color based on confidence
                if score >= 0.7:
                    color = 'red'
                elif score >= 0.4:
                    color='orange' 
                else:
                    color = 'yellow'
                
                # Create corners
                corners = np.array([
                    [-l/2, -w/2], [l/2, -w/2],
                    [l/2, w/2], [-l/2, w/2], [-l/2, -w/2]
                ])
                
                # Rotate
                rot_mat = np.array([
                    [np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw), np.cos(yaw)]
                ])
                corners = corners @ rot_mat.T + np.array([x, y])
                
                # Plot box
                ax.plot(corners[:, 0], corners[:, 1], 
                       linewidth=2, color=color)
                
                # Add score label
                ax.text(x, y, f'{score:.2f}',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor=color, alpha=0.9),
                       fontsize=10, color='white', weight='bold',
                       ha='center', va='center')
    
    ax.set_xlabel('X (m)', fontsize=14, weight='bold')
    ax.set_ylabel('Y (m)', fontsize=14, weight='bold')
    ax.set_title(title, fontsize=16, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: {output_path}")

print("\n" + "="*70)
print("  GENERATING BEV DETECTION VISUALIZATIONS")
print("="*70)

# Process KITTI dataset
print("\n[1/2] Processing KITTI PointPillars Detections...")
print("-" * 70)

kitti_bins = sorted(glob.glob('data/kitti/training/velodyne/*.bin'))
kitti_preds = 'outputs/kitti_pointpillars/preds'
kitti_output = 'results/bev_visualizations/kitti'
os.makedirs(kitti_output, exist_ok=True)

if kitti_bins:
    for i, bin_file in enumerate(kitti_bins[:15]):  # Process 15 samples
        sample_id = Path(bin_file).stem
        pred_file = f"{kitti_preds}/{sample_id}.json"
        
        if os.path.exists(pred_file):
            points = load_kitti_points(bin_file)
            preds = load_predictions(pred_file)
            
            output = f"{kitti_output}/kitti_pointpillars_{sample_id}_bev.png"
            create_bev_plot(
                points, preds,
                f"PointPillars - 3D Object Detection (BEV) - KITTI Sample {sample_id}",
                output
            )
else:
    print("  ⚠ No KITTI .bin files found")

# Process nuScenes dataset  
print("\n[2/2] Processing nuScenes PointPillars Detections...")
print("-" * 70)

nuscenes_bin = 'data/nuscenes_demo/lidar/sample.pcd.bin'
nuscenes_preds = sorted(glob.glob('outputs/nuscenes_pointpillars/preds/*.json'))
nuscenes_output = 'results/bev_visualizations/nuscenes'
os.makedirs(nuscenes_output, exist_ok=True)

if os.path.exists(nuscenes_bin) and nuscenes_preds:
    points = load_kitti_points(nuscenes_bin)
    
    for i, pred_file in enumerate(nuscenes_preds[:15]):  # Process 15 samples
        sample_id = Path(pred_file).stem
        preds = load_predictions(pred_file)
        
        output = f"{nuscenes_output}/nuscenes_pointpillars_{i:03d}_bev.png"
        create_bev_plot(
            points, preds,
            f"PointPillars - 3D Object Detection (BEV) - nuScenes Sample {i}",
            output
        )
else:
    print("  ⚠ nuScenes data not fully available")

print("\n" + "="*70)
print("  ✓ BEV VISUALIZATION GENERATION COMPLETE!")
print("="*70)
print(f"\nOutput locations:")
print(f"  • KITTI:    {kitti_output}/")
print(f"  • nuScenes: {nuscenes_output}/")
print("\n")
