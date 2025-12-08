import sys
sys.path.insert(0, '/teamspace/studios/this_studio/3D-object-detection')
from mmdet3d.apis import init_model, inference_detector
import os
import numpy as np

print("="*60)
print("POINTPILLARS KITTI INFERENCE")
print("="*60)

# Paths
config_url = 'https://raw.githubusercontent.com/open-mmlab/mmdetection3d/main/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
checkpoint = '/teamspace/studios/this_studio/3d_detection_workspace/checkpoints/pointpillars_kitti.pth'

# Create synthetic demo data if none exists
pcd_dir = '/teamspace/studios/this_studio/3d_detection_workspace/data/kitti/training/velodyne'
os.makedirs(pcd_dir, exist_ok=True)

pcd_file = os.path.join(pcd_dir, '000008.bin')
if not os.path.exists(pcd_file):
    print(f"Creating synthetic point cloud data at {pcd_file}")
    # Create minimal synthetic point cloud (N x 4: x,y,z,intensity)
    points = np.random.randn(1000, 4).astype(np.float32)
    points[:, :3] *= 10  # Scale xyz
    points[:, 3] = np.abs(points[:, 3])  # Positive intensity
    points.tofile(pcd_file)
    print(f"✅ Created synthetic data: {points.shape}")

print(f"\nLoading PointPillars model...")
model = init_model(config_url, checkpoint, device='cpu')
print("✅ Model loaded successfully!")

print(f"\nRunning inference on {pcd_file}...")
result, data = inference_detector(model, pcd_file)

print(f"\n{'='*60}")
print("DETECTION RESULTS:")
print(f"{'='*60}")
print(f"Number of detections: {len(result.pred_instances_3d.bboxes_3d)}")
if len(result.pred_instances_3d.bboxes_3d) > 0:
    print(f"Bounding boxes shape: {result.pred_instances_3d.bboxes_3d.tensor.shape}")
    print(f"Scores: {result.pred_instances_3d.scores_3d[:5]}...")  # First 5
    print(f"Labels: {result.pred_instances_3d.labels_3d[:5]}...")  # First 5
else:
    print("No objects detected (expected for synthetic data)")

print(f"\n✅ PointPillars inference complete!\n")
