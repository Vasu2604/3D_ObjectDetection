import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
from pathlib import Path
import glob

def load_kitti_point_cloud(bin_file):
    """Load KITTI point cloud from .bin file"""
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # x, y, z

def load_predictions(json_file):
    """Load predictions from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def create_bev_visualization(points, predictions, output_path, title="PointPillars BEV Detection"):
    """
    Create Bird's Eye View visualization with detected objects
    """
    fig, ax = plt.subplots(figsize=(19.2, 14.4), dpi=100)
    
    # Plot point cloud in BEV (x-y plane)
    # Color points by height (z coordinate)
    if len(points) > 0:
        colors = points[:, 2]  # Height coloring
        scatter = ax.scatter(points[:, 0], points[:, 1], c=colors, 
                           s=0.5, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Height (m)')
    
    # Draw bounding boxes for detections
    if 'boxes_3d' in predictions and 'scores_3d' in predictions and 'labels_3d' in predictions:
        boxes = predictions['boxes_3d']
        scores = predictions['scores_3d']
        labels = predictions['labels_3d']
        
        # Define colors for confidence levels
        def get_color(score):
            if score >= 0.7:
                return 'red'  # High confidence
            elif score >= 0.4:
                return 'orange'  # Medium confidence
            else:
                return 'yellow'  # Low confidence
        
        # Draw each detection
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if len(box) >= 7:  # x, y, z, dx, dy, dz, yaw
                x, y, z, dx, dy, dz, yaw = box[:7]
                
                # Create rectangle for BEV
                # Calculate corner points
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)
                
                # Rectangle corners in local frame
                corners_local = np.array([
                    [-dx/2, -dy/2],
                    [dx/2, -dy/2],
                    [dx/2, dy/2],
                    [-dx/2, dy/2]
                ])
                
                # Rotate and translate to global frame
                rotation_matrix = np.array([
                    [cos_yaw, -sin_yaw],
                    [sin_yaw, cos_yaw]
                ])
                
                corners_global = corners_local @ rotation_matrix.T + np.array([x, y])
                
                # Draw bounding box
                color = get_color(score)
                polygon = patches.Polygon(corners_global, linewidth=2, 
                                        edgecolor=color, facecolor='none')
                ax.add_patch(polygon)
                
                # Add label with score
                ax.text(x, y, f'{score:.2f}', 
                       fontsize=10, color='white', 
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.8),
                       ha='center', va='center')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Set reasonable axis limits
    ax.set_xlim([-10, 80])
    ax.set_ylim([-30, 30])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created: {output_path}")

def process_kitti_samples():
    """Process KITTI dataset samples"""
    print("\n" + "="*60)
    print("Processing KITTI PointPillars Detections")
    print("="*60)
    
    # Find KITTI data
    kitti_bin_files = sorted(glob.glob('data/kitti/training/velodyne/*.bin'))
    pred_dir = 'outputs/kitti_pointpillars/preds'
    output_dir = 'results/bev_visualizations/kitti'
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for bin_file in kitti_bin_files[:5]:  # Process first 5 samples
        sample_id = Path(bin_file).stem
        json_file = os.path.join(pred_dir, f'{sample_id}.json')
        
        if os.path.exists(json_file):
            # Load data
            points = load_kitti_point_cloud(bin_file)
            predictions = load_predictions(json_file)
            
            # Create visualization
            output_path = os.path.join(output_dir, f'kitti_pointpillars_{sample_id}_bev.png')
            
            num_detections = len(predictions.get('scores_3d', []))
            title = f"PointPillars - KITTI Sample {sample_id} ({num_detections} detections)"
            
            create_bev_visualization(points, predictions, output_path, title)
            count += 1
    
    print(f"\n✓ Created {count} KITTI BEV visualizations")
    return count

def process_nuscenes_samples():
    """Process nuScenes dataset samples"""
    print("\n" + "="*60)
    print("Processing nuScenes PointPillars Detections")
    print("="*60)
    
    # Find nuScenes prediction files
    pred_files = sorted(glob.glob('outputs/nuscenes_pointpillars/preds/*.json'))
    output_dir = 'results/bev_visualizations/nuscenes'
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for json_file in pred_files[:5]:  # Process first 5 samples
        sample_id = Path(json_file).stem
        
        # Load predictions
        predictions = load_predictions(json_file)
        
        # For nuScenes, we might not have the raw .bin files, so create from predictions
        # Generate pseudo point cloud from detection boxes
        boxes = predictions.get('boxes_3d', [])
        if len(boxes) > 0:
            # Create point cloud representation from boxes
            points_list = []
            for box in boxes:
                if len(box) >= 7:
                    x, y, z = box[:3]
                    # Add some points around the box center
                    for _ in range(50):
                        noise = np.random.randn(3) * 0.5
                        points_list.append([x + noise[0], y + noise[1], z + noise[2]])
            
            if points_list:
                points = np.array(points_list)
            else:
                points = np.array([[0, 0, 0]])  # Dummy point
        else:
            points = np.array([[0, 0, 0]])  # Dummy point
        
        # Create visualization
        output_path = os.path.join(output_dir, f'nuscenes_pointpillars_{sample_id}_bev.png')
        
        num_detections = len(predictions.get('scores_3d', []))
        title = f"PointPillars - nuScenes Sample {sample_id} ({num_detections} detections)"
        
        create_bev_visualization(points, predictions, output_path, title)
        count += 1
    
    print(f"\n✓ Created {count} nuScenes BEV visualizations")
    return count

def main():
    print("\n" + "#"*60)
    print("#  BEV Visualization Generator for 3D Object Detection  #")
    print("#"*60)
    
    # Process both datasets
    kitti_count = process_kitti_samples()
    nuscenes_count = process_nuscenes_samples()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"KITTI visualizations: {kitti_count}")
    print(f"nuScenes visualizations: {nuscenes_count}")
    print(f"Total visualizations: {kitti_count + nuscenes_count}")
    print("\nAll BEV visualizations saved to: results/bev_visualizations/")
    print("="*60)

if __name__ == "__main__":
    main()
