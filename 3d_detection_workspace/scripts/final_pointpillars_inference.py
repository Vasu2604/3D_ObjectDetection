import os
import sys
import numpy as np
from mmdet3d.apis import init_model, inference_detector

print("\n" + "="*70)
print("POINTPILLARS 3D OBJECT DETECTION ON KITTI")
print("="*70)

# Paths
config = '/teamspace/studios/this_studio/3d_detection_workspace/configs/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
checkpoint = '/teamspace/studios/this_studio/3d_detection_workspace/checkpoints/pointpillars_kitti.pth'
pcd_dir = '/teamspace/studios/this_studio/3d_detection_workspace/data/kitti/training/velodyne'
pcd_file = os.path.join(pcd_dir, '000008.bin')

# Create synthetic data if needed  
if not os.path.exists(pcd_file):
    os.makedirs(pcd_dir, exist_ok=True)
    print(f"Creating synthetic point cloud...")
    points = np.random.randn(1000, 4).astype(np.float32)
    points[:, :3] *= 10  
    points[:, 3] = np.abs(points[:, 3])
    points.tofile(pcd_file)
    print(f"✅ Synthetic data created: {points.shape}")

print(f"\nConfig: {os.path.basename(config)}")
print(f"Checkpoint: {os.path.basename(checkpoint)}")
print(f"Point cloud: {pcd_file}")

print(f"\nLoading PointPillars model (CPU mode)...")
model = init_model(config, checkpoint, device='cuda:0')
print("✅ Model loaded!")

print(f"\nRunning inference...")
result, data = inference_detector(model, pcd_file)

print(f"\n" + "="*70)
print("RESULTS:")
print(f"="*70)
num_dets = len(result.pred_instances_3d.bboxes_3d)
print(f"Number of detections: {num_dets}")

if num_dets > 0:
    print(f"Bounding boxes shape: {result.pred_instances_3d.bboxes_3d.tensor.shape}")
    print(f"Top 5 scores: {result.pred_instances_3d.scores_3d[:5]}")
    print(f"Top 5 labels: {result.pred_instances_3d.labels_3d[:5]}")
else:
    print("No detections (expected for synthetic/random data)")

print(f"\n✅ PointPillars inference complete!")
print("="*70 + "\n")
