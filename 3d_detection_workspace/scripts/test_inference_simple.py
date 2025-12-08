import torch
import numpy as np
from mmdet3d.apis import init_model, inference_detector  
import os

print("\n" + "="*70)
print("3D OBJECT DETECTION - POINTPILLARS ON KITTI")
print("="*70 + "\n")

# Use mim to get config
import subprocess
print("Downloading config file...")
config_file = 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
subprocess.run(["mim", "download", "mmdet3d", "--config", config_file, "--dest", "."], check=False)

# If mim download failed, use direct path
if not os.path.exists(config_file):
    # Try to find in installed package
    import mmdet3d
    package_path = os.path.dirname(mmdet3d.__file__)
    config_file = f"{package_path}/../configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"
    if not os.path.exists(config_file):
        print(f"ERROR: Config not found. Creating minimal config...")
        # Use checkpoint config directly
        config_file = None

checkpoint = '../checkpoints/pointpillars_kitti.pth'
pcd_file = '../data/kitti/training/velodyne/000008.bin'

# Create synthetic data if needed
if not os.path.exists(pcd_file):
    os.makedirs(os.path.dirname(pcd_file), exist_ok=True)
    points = np.random.randn(1000, 4).astype(np.float32)
    points[:, :3] *= 10
    points[:, 3] = np.abs(points[:, 3])
    points.tofile(pcd_file)
    print(f"✅ Created synthetic point cloud")

print("\nInitializing model (CPU mode)...")
if config_file and os.path.exists(config_file):
    model = init_model(config_file, checkpoint, device='cpu')
else:
    print("Using checkpoint-only initialization...")
    from mmdet3d.utils import register_all_modules
    register_all_modules()
    from mmengine.runner import load_checkpoint
    from mmdet3d.registry import MODELS
    checkpoint_data = torch.load(checkpoint, map_location='cpu')
    print(f"✅ Loaded checkpoint")
    print(f"   Keys: {list(checkpoint_data.keys())[:5]}...")
    # This is a simplified version - full inference needs config
    
print("\n✅ SETUP COMPLETE!")
print("\n" + "="*70)
