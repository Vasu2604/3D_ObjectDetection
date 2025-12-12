import cv2
import os
import glob
from pathlib import Path
import imageio

def create_video_from_images(image_folder, output_video, fps=2):
    """
    Create a video from PNG images in a folder
    """
    # Get all PNG files
    images = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    
    if not images:
        print(f"No PNG images found in {image_folder}")
        return False
    
    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    if frame is None:
        print(f"Could not read image: {images[0]}")
        return False
    
    height, width, layers = frame.shape
    
    # Create video writer for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"Creating video from {len(images)} images...")
    
    for image_path in images:
        frame = cv2.imread(image_path)
        if frame is not None:
            video.write(frame)
    
    video.release()
    cv2.destroyAllWindows()
    
    print(f"Video saved to: {output_video}")
    return True

def create_gif_from_images(image_folder, output_gif, fps=2):
    """
    Create a GIF from PNG images in a folder
    """
    images = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    
    if not images:
        print(f"No PNG images found in {image_folder}")
        return False
    
    print(f"Creating GIF from {len(images)} images...")
    
    frames = []
    for image_path in images:
        frames.append(imageio.imread(image_path))
    
    imageio.mimsave(output_gif, frames, fps=fps)
    print(f"GIF saved to: {output_gif}")
    return True

def main():
    # Define PointPillars output directories
    pointpillars_dirs = [
        ('outputs/kitti_pointpillars', 'results/kitti_pointpillars_video.mp4', 'results/kitti_pointpillars_video.gif'),
        ('outputs/kitti_pointpillars_gpu', 'results/kitti_pointpillars_gpu_video.mp4', 'results/kitti_pointpillars_gpu_video.gif'),
        ('outputs/kitti_pointpillars_cuda', 'results/kitti_pointpillars_cuda_video.mp4', 'results/kitti_pointpillars_cuda_video.gif'),
        ('outputs/kitti_pointpillars_3class', 'results/kitti_pointpillars_3class_video.mp4', 'results/kitti_pointpillars_3class_video.gif'),
        ('outputs/nuscenes_pointpillars', 'results/nuscenes_pointpillars_video.mp4', 'results/nuscenes_pointpillars_video.gif'),
    ]
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    for image_dir, video_path, gif_path in pointpillars_dirs:
        if os.path.exists(image_dir):
            print(f"\n{'='*60}")
            print(f"Processing: {image_dir}")
            print(f"{'='*60}")
            
            # Create MP4 video
            if create_video_from_images(image_dir, video_path, fps=2):
                print(f"✓ MP4 video created successfully")
            else:
                print(f"✗ Failed to create MP4 video")
            
            # Create GIF
            if create_gif_from_images(image_dir, gif_path, fps=2):
                print(f"✓ GIF created successfully")
            else:
                print(f"✗ Failed to create GIF")
        else:
            print(f"\nSkipping {image_dir} - directory not found")
    
    print(f"\n{'='*60}")
    print("Video creation complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
