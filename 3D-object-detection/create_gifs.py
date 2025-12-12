import glob
import os
from PIL import Image

def create_gif_from_images(image_folder, output_gif, fps=2, max_frames=10):
    """
    Create a GIF from PNG images using PIL
    """
    images = sorted(glob.glob(os.path.join(image_folder, '*.png')))[:max_frames]
    
    if not images:
        print(f"No PNG images found in {image_folder}")
        return False
    
    print(f"Creating GIF from {len(images)} images...")
    
    frames = []
    for image_path in images:
        img = Image.open(image_path)
        # Resize for smaller file size
        img = img.resize((img.width // 2, img.height // 2), Image.Resampling.LANCZOS)
        frames.append(img)
    
    # Save as GIF
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000/fps),
        loop=0
    )
    
    print(f"GIF saved to: {output_gif}")
    return True

def main():
    pointpillars_dirs = [
        ('outputs/kitti_pointpillars', 'results/kitti_pointpillars_video.gif'),
        ('outputs/kitti_pointpillars_gpu', 'results/kitti_pointpillars_gpu_video.gif'),
        ('outputs/kitti_pointpillars_cuda', 'results/kitti_pointpillars_cuda_video.gif'),
        ('outputs/kitti_pointpillars_3class', 'results/kitti_pointpillars_3class_video.gif'),
        ('outputs/nuscenes_pointpillars', 'results/nuscenes_pointpillars_video.gif'),
    ]
    
    for image_dir, gif_path in pointpillars_dirs:
        if os.path.exists(image_dir):
            print(f"\n{'='*60}")
            print(f"Processing: {image_dir}")
            print(f"{'='*60}")
            
            if create_gif_from_images(image_dir, gif_path, fps=2, max_frames=10):
                print(f"✓ GIF created successfully")
            else:
                print(f"✗ Failed to create GIF")
        else:
            print(f"\nSkipping {image_dir} - directory not found")
    
    print(f"\n{'='*60}")
    print("GIF creation complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
