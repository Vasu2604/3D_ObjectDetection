import os
import sys
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from mmdet3d.apis import init_model, inference_detector
from mmdet3d.registry import VISUALIZERS
import mmcv
from mmengine import Config

def save_detection_results(result, pcd_file, output_dir, model_name, dataset_name):
    """Save detection results as .json metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract results
    bboxes = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
    scores = result.pred_instances_3d.scores_3d.cpu().numpy()
    labels = result.pred_instances_3d.labels_3d.cpu().numpy()
    
    # Create metadata
    metadata = {
        'model': model_name,
        'dataset': dataset_name,
        'point_cloud_file': pcd_file,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_detections': len(bboxes),
        'detections': []
    }
    
    for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
        detection = {
            'id': i,
            'bbox_3d': bbox.tolist(),
            'confidence': float(score),
            'class_id': int(label),
            'class_name': 'Car' if label == 0 else f'Class_{label}'
        }
        metadata['detections'].append(detection)
    
    # Save JSON
    json_path = os.path.join(output_dir, 'metadata', 'detections.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to {json_path}")
    return metadata

def save_visualization_image(result, pcd_file, output_dir, model_name):
    """Save 2D bird's eye view visualization as .png"""
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # Load point cloud
    points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
    
    # Create BEV visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Plot point cloud (BEV)
    ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=0.1, cmap='viridis', alpha=0.5)
    
    # Plot bounding boxes
    bboxes = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
    scores = result.pred_instances_3d.scores_3d.cpu().numpy()
    
    for bbox, score in zip(bboxes, scores):
        # Extract corners for BEV (x, y)
        x, y, z, dx, dy, dz, yaw = bbox[:7]
        
        # Simple box corners in BEV
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        corners_x = [x + dx/2*cos_yaw - dy/2*sin_yaw,
                     x + dx/2*cos_yaw + dy/2*sin_yaw,
                     x - dx/2*cos_yaw + dy/2*sin_yaw,
                     x - dx/2*cos_yaw - dy/2*sin_yaw,
                     x + dx/2*cos_yaw - dy/2*sin_yaw]
        corners_y = [y + dx/2*sin_yaw + dy/2*cos_yaw,
                     y + dx/2*sin_yaw - dy/2*cos_yaw,
                     y - dx/2*sin_yaw - dy/2*cos_yaw,
                     y - dx/2*sin_yaw + dy/2*cos_yaw,
                     y + dx/2*sin_yaw + dy/2*cos_yaw]
        
        color = 'r' if score > 0.5 else 'y'
        ax.plot(corners_x, corners_y, color=color, linewidth=2, label=f'Score: {score:.2f}')
        ax.text(x, y, f'{score:.2f}', color='white', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{model_name} - 3D Object Detection (BEV)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Save
    img_path = os.path.join(output_dir, 'images', 'detection_bev.png')
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved BEV image to {img_path}")
    return img_path

def save_point_cloud_with_boxes(result, pcd_file, output_dir):
    """Save point cloud with bounding boxes as .ply"""
    os.makedirs(os.path.join(output_dir, 'pointclouds'), exist_ok=True)
    
    # Load points
    points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
    
    # Get bboxes
    bboxes = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
    
    # Create simple PLY content
    ply_path = os.path.join(output_dir, 'pointclouds', 'detection_result.ply')
    
    with open(ply_path, 'w') as f:
        # PLY header
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        
        # Write points
        for pt in points:
            x, y, z, intensity = pt
            color = int(intensity * 255) if intensity <= 1.0 else int(intensity)
            f.write(f'{x} {y} {z} {color} {color} {color}\n')
    
    print(f"✓ Saved point cloud to {ply_path}")
    return ply_path

def run_inference_with_saving(config_file, checkpoint_file, pcd_file, 
                               output_base, model_name, dataset_name):
    """Run inference and save all artifacts"""
    
    print(f"\n{'='*70}")
    print(f"Running {model_name} on {dataset_name}")
    print(f"{'='*70}")
    
    # Initialize model
    print(f"Loading model from {checkpoint_file}...")
    model = init_model(config_file, checkpoint_file, device='cpu')
    print("✓ Model loaded!")
    
    # Run inference
    print(f"\nRunning inference on {pcd_file}...")
    result, data = inference_detector(model, pcd_file)
    
    # Extract results
    bboxes = result.pred_instances_3d.bboxes_3d.tensor
    scores = result.pred_instances_3d.scores_3d
    labels = result.pred_instances_3d.labels_3d
    
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"Number of detections: {len(bboxes)}")
    print(f"Top 5 scores: {scores[:5].cpu().numpy()}")
    print(f"Top 5 labels: {labels[:5].cpu().numpy()}")
    
    # Save all artifacts
    output_dir = os.path.join(output_base, dataset_name, model_name)
    
    print(f"\nSaving artifacts to {output_dir}...")
    metadata = save_detection_results(result, pcd_file, output_dir, model_name, dataset_name)
    img_path = save_visualization_image(result, pcd_file, output_dir, model_name)
    ply_path = save_point_cloud_with_boxes(result, pcd_file, output_dir)
    
    print(f"\n✓ All artifacts saved successfully!")
    print(f"{'='*70}\n")
    
    return result, metadata

if __name__ == '__main__':
    # Configuration
    BASE_DIR = '/teamspace/studios/this_studio/3d_detection_workspace'
    OUTPUT_BASE = os.path.join(BASE_DIR, 'results')
    
    configs_and_checkpoints = [
        {
            'name': 'pointpillars',
            'config': os.path.join(BASE_DIR, 'configs/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'),
            'checkpoint': os.path.join(BASE_DIR, 'checkpoints/pointpillars_kitti.pth')
        },
        {
            'name': 'second',
            'config': os.path.join(BASE_DIR, 'configs/second_hv_secfpn_8xb6-80e_kitti-3d-car.py'),
            'checkpoint': os.path.join(BASE_DIR, 'checkpoints/second_kitti.pth')
        }
    ]
    
    datasets = [
        {'name': 'kitti', 'file': os.path.join(BASE_DIR, 'data/kitti/training/velodyne/000008.bin')},
        {'name': 'nuscenes', 'file': os.path.join(BASE_DIR, 'data/nuscenes/sample_lidar.bin')}
    ]
    
    # Run all combinations
    all_results = []
    
    for model_cfg in configs_and_checkpoints:
        for dataset in datasets:
            result, metadata = run_inference_with_saving(
                config_file=model_cfg['config'],
                checkpoint_file=model_cfg['checkpoint'],
                pcd_file=dataset['file'],
                output_base=OUTPUT_BASE,
                model_name=model_cfg['name'],
                dataset_name=dataset['name']
            )
            all_results.append({
                'model': model_cfg['name'],
                'dataset': dataset['name'],
                'metadata': metadata
            })
    
    print("\n" + "="*70)
    print("ALL INFERENCE COMPLETE!")
    print("="*70)
    print(f"\nProcessed {len(all_results)} model-dataset combinations")
    print(f"Results saved to: {OUTPUT_BASE}")
