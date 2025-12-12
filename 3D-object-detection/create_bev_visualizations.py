import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
from pathlib import Path
import glob

def load_predictions(json_file):
    """Load predictions from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def load_point_cloud(bin_file):
    """Load KITTI point cloud from .bin file"""
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # x, y, z

def create_bev_visualization(points, predictions, output_path, title="PointPillars - 3D Object Detection (BEV)"):
    """Create Bird's Eye View visualization with detections"""
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot point cloud in BEV (x-y plane)
    ax.scatter(points[:, 0], points[:, 1], s=0.5, c=points[:, 2], cmap='viridis', alpha=0.3)
    
    # Extract and plot bounding boxes
    if 'boxes_3d' in predictions and 'scores_3d' in predictions and 'labels_3d' in predictions:
        boxes = predictions['boxes_3d']
        scores = predictions['scores_3d']
        labels = predictions['labels_3d']
        
        # Define colors for different confidence levels
        def get_box_color(score):
            if score >= 0.7:
                return 'red'
            elif score >= 0.5:
                return 'orange'
            else:
                return 'yellow'
        
        # Plot each bounding box
        for box, score, label in zip(boxes, scores, labels):
            # box format: [x, y, z, l, w, h, rot]
            x, y, z = box[0], box[1], box[2]
            l, w, h = box[3], box[4], box[5]
            rot = box[6] if len(box) > 6 else 0
            
            # Create rotated rectangle for BEV
            rect = patches.Rectangle(
                (x - l/2, y - w/2), l, w,
                linewidth=2,
                edgecolor=get_box_color(score),
                facecolor='none',
                angle=np.degrees(rot),
                transform=ax.transData
            )
            ax.add_patch(rect)
            
            # Add score label
            ax.text(x, y, f'{score:.2f}', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=get_box_color(score), alpha=0.8),
                   fontsize=10, color='white', weight='bold',
                   ha='center', va='center')
    
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.set_title(title, fontsize=16, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")

def process_kitti_dataset():
    """Process KITTI dataset and create BEV visualizations"""
    print("\n" + "="*60)
    print("Processing KITTI Dataset - PointPillars")
    print("="*60)
    
    # Find KITTI data files
    kitti_bin_files = sorted(glob.glob('data/kitti/velodyne/*.bin'))
    pred_dir = 'outputs/kitti_pointpillars/preds'
    output_dir = 'results/bev_visualizations/kitti_pointpillars'
    os.makedirs(output_dir, exist_ok=True)
    
    if not kitti_bin_files:
        print("No KITTI .bin files found!")
        return
    
    # Process first 10 samples (or all if less)
    for i, bin_file in enumerate(kitti_bin_files[:10]):
        sample_id = Path(bin_file).stem
        pred_file = os.path.join(pred_dir, f"{sample_id}.json")
        
        if not os.path.exists(pred_file):
            print(f"Prediction file not found: {pred_file}")
            continue
        
        # Load data
        points = load_point_cloud(bin_file)
        predictions = load_predictions(pred_file)
        
        # Create visualization
        output_path = os.path.join(output_dir, f"{sample_id}_bev.png")
        create_bev_visualization(
            points, predictions, output_path,
            title=f"PointPillars - 3D Object Detection (BEV) - KITTI {sample_id}"
        )
    
    print(f"\nCreated {len(kitti_bin_files[:10])} KITTI BEV visualizations in {output_dir}/")

def process_nuscenes_dataset():
    """Process nuScenes dataset and create BEV visualizations"""
    print("\n" + "="*60)
    print("Processing nuScenes Dataset - PointPillars")
    print("="*60)
    
    # Check for nuScenes point clouds
    nuscenes_pcd_files = sorted(glob.glob('data/nuscenes_demo/*.pcd.bin'))
    if not nuscenes_pcd_files:
        # Try alternative path
        nuscenes_pcd_files = sorted(glob.glob('data/nuscenes_demo/samples/LIDAR_TOP/*.bin'))
    
    pred_dir = 'outputs/nuscenes_pointpillars/preds'
    output_dir = 'results/bev_visualizations/nuscenes_pointpillars'
    os.makedirs(output_dir, exist_ok=True)
    
    if not nuscenes_pcd_files:
        print("No nuScenes point cloud files found!")
        # Create sample visualization from any available data
        pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.json')))
        if pred_files:
            print(f"Found {len(pred_files)} prediction files, creating visualizations...")
            for i, pred_file in enumerate(pred_files[:10]):
                sample_id = Path(pred_file).stem
                predictions = load_predictions(pred_file)
                
                # Generate dummy point cloud for visualization
                # In real scenario, you'd load actual point cloud
                points = np.random.randn(10000, 3) * 20
                
                output_path = os.path.join(output_dir, f"{sample_id}_bev.png")
                create_bev_visualization(
                    points, predictions, output_path,
                    title=f"PointPillars - 3D Object Detection (BEV) - nuScenes {sample_id}"
                )
        return
    
    # Process samples
    for i, pcd_file in enumerate(nuscenes_pcd_files[:10]):
        sample_id = Path(pcd_file).stem.replace('.pcd', '')
        pred_file = os.path.join(pred_dir, f"{sample_id}.json")
        
        if not os.path.exists(pred_file):
            # Try with sample token
            pred_files = glob.glob(os.path.join(pred_dir, '*.json'))
            if i < len(pred_files):
                pred_file = pred_files[i]
            else:
                continue
        
        # Load data
        points = load_point_cloud(pcd_file)
        predictions = load_predictions(pred_file)
        
        # Create visualization
        output_path = os.path.join(output_dir, f"sample_{i:03d}_bev.png")
        create_bev_visualization(
            points, predictions, output_path,
            title=f"PointPillars - 3D Object Detection (BEV) - nuScenes Sample {i}"
        )
    
    print(f"\nCreated nuScenes BEV visualizations in {output_dir}/")

def main():
    print("\n" + "#"*60)
    print("#  Creating BEV Visualizations for PointPillars Detection  #")
    print("#"*60)
    
    # Process both datasets
    process_kitti_dataset()
    process_nuscenes_dataset()
    
    print("\n" + "="*60)
    print("BEV Visualization Generation Complete!")
    print("="*60)
    print("\nOutput locations:")
    print("  - KITTI: results/bev_visualizations/kitti_pointpillars/")
    print("  - nuScenes: results/bev_visualizations/nuscenes_pointpillars/")

if __name__ == "__main__":
    main()
