import open3d as o3d
import glob
import os
from pathlib import Path
import numpy as np

def create_screenshots_from_ply(ply_file, output_dir, basename):
    """
    Create screenshots from a PLY file showing different views
    """
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(ply_file)
    
    if len(pcd.points) == 0:
        print(f"Empty point cloud: {ply_file}")
        return False
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)
    vis.add_geometry(pcd)
    
    # Set viewing options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    
    # Get view control
    ctr = vis.get_view_control()
    
    # Different camera angles
    views = [
        {'name': 'front', 'params': None},
        {'name': 'top', 'params': {'front': [0, 0, -1], 'lookat': [0, 0, 0], 'up': [0, -1, 0], 'zoom': 0.5}},
        {'name': 'side', 'params': {'front': [-1, 0, 0], 'lookat': [0, 0, 0], 'up': [0, 0, 1], 'zoom': 0.5}},
    ]
    
    screenshots_created = []
    
    for view in views:
        view_name = view['name']
        
        # Reset view
        ctr.reset_camera_local_rotate()
        
        # Render
        vis.poll_events()
        vis.update_renderer()
        
        # Save screenshot
        screenshot_path = os.path.join(output_dir, f"{basename}_{view_name}.png")
        vis.capture_screen_image(screenshot_path, do_render=True)
        screenshots_created.append(screenshot_path)
        print(f"  Created: {screenshot_path}")
    
    vis.destroy_window()
    return screenshots_created

def main():
    # Process PointPillars PLY files
    ply_dirs = [
        ('outputs/kitti_pointpillars', 'results/screenshots/kitti_pointpillars'),
        ('outputs/kitti_pointpillars_gpu', 'results/screenshots/kitti_pointpillars_gpu'),
        ('outputs/kitti_pointpillars_cuda', 'results/screenshots/kitti_pointpillars_cuda'),
        ('outputs/kitti_pointpillars_3class', 'results/screenshots/kitti_pointpillars_3class'),
        ('outputs/nuscenes_pointpillars', 'results/screenshots/nuscenes_pointpillars'),
        ('outputs/3dssd', 'results/screenshots/3dssd'),
        ('outputs/nuscenes_centerpoint', 'results/screenshots/nuscenes_centerpoint'),
    ]
    
    print("\n" + "="*60)
    print("Creating Open3D screenshots from PLY files")
    print("="*60)
    
    total_screenshots = 0
    
    for input_dir, output_dir in ply_dirs:
        if not os.path.exists(input_dir):
            print(f"\nSkipping {input_dir} - directory not found")
            continue
        
        # Find PLY files
        ply_files = glob.glob(os.path.join(input_dir, '*.ply'))
        
        if not ply_files:
            print(f"\nNo PLY files found in {input_dir}")
            continue
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process first 3 PLY files (to keep it manageable)
        ply_files = sorted(ply_files)[:3]
        
        print(f"\n{input_dir}: Processing {len(ply_files)} PLY files")
        
        for ply_file in ply_files:
            basename = Path(ply_file).stem
            print(f"  Processing: {basename}")
            
            try:
                screenshots = create_screenshots_from_ply(ply_file, output_dir, basename)
                if screenshots:
                    total_screenshots += len(screenshots)
            except Exception as e:
                print(f"  Error processing {ply_file}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Screenshot creation complete! Created {total_screenshots} screenshots.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
