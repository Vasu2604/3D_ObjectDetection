import os
import sys
import numpy as np
from mmdet3d.apis import init_model, inference_detector

print("\n" + "="*70)
print("SECOND 3D OBJECT DETECTION ON KITTI")
print("="*70)

# Paths
config = '/teamspace/studios/this_studio/3d_detection_workspace/configs/second_hv_secfpn_8xb6-80e_kitti-3d-car.py'
checkpoint = '/teamspace/studios/this_studio/3d_detection_workspace/checkpoints/second_kitti.pth'
pcd_file = '/teamspace/studios/this_studio/3d_detection_workspace/data/kitti/training/velodyne/000008.bin'

if not os.path.exists(pcd_file):
    print(f"ERROR: Point cloud not found: {pcd_file}")
    sys.exit(1)

print(f"\nConfig: {os.path.basename(config)}")
print(f"Checkpoint: {os.path.basename(checkpoint)}")
print(f"Point cloud: {pcd_file}")

print(f"\nLoading SECOND model (CPU mode)...")
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

print(f"\n✅ SECOND inference complete!")
print("="*70 + "\n")
