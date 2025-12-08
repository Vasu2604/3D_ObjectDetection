import sys
sys.path.insert(0, '/teamspace/studios/this_studio/3D-object-detection')
from mmdet3d.apis import init_model, inference_detector
import os
import numpy as np

print("="*60)
print("SECOND KITTI INFERENCE")
print("="*60)

# Paths  
config_url = 'https://raw.githubusercontent.com/open-mmlab/mmdetection3d/main/configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py'
checkpoint = '/teamspace/studios/this_studio/3d_detection_workspace/checkpoints/second_kitti.pth'
pcd_file = '/teamspace/studios/this_studio/3d_detection_workspace/data/kitti/training/velodyne/000008.bin'

if not os.path.exists(pcd_file):
    print(f"ERROR: Point cloud file not found: {pcd_file}")
    sys.exit(1)

print(f"\nLoading SECOND model...")
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

print(f"\n✅ SECOND inference complete!\n")
