import cv2
import numpy as np
from pathlib import Path
import glob

print("="*80)
print("Creating CenterPoint Demo Videos")
print("="*80)

KITTI_IMG_DIR = Path("results/centerpoint_kitti_bev")
NUSCENES_IMG_DIR = Path("results/centerpoint_nuscenes_bev")

def create_video(image_dir, output_video, fps=3):
    images = sorted(glob.glob(str(image_dir / "*.png")))
    
    if not images:
        print(f"No images found in {image_dir}")
        return False
    
    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    if frame is None:
        return False
    
    height, width, _ = frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    print(f"Creating video from {len(images)} images...")
    
    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is not None:
            video.write(frame)
    
    video.release()
    print(f"✓ Video saved: {output_video}")
    return True

print("\n[1/2] Creating KITTI CenterPoint video...")
kitti_video = Path("results/centerpoint_kitti_demo.mp4")
create_video(KITTI_IMG_DIR, kitti_video, fps=3)

print("\n[2/2] Creating nuScenes CenterPoint video...")
nuscenes_video = Path("results/centerpoint_nuscenes_demo.mp4")
create_video(NUSCENES_IMG_DIR, nuscenes_video, fps=3)

print("\n" + "="*80)
print("CenterPoint Demo Videos Complete!")
print("="*80)
print(f"KITTI video: {kitti_video}")
print(f"nuScenes video: {nuscenes_video}")
