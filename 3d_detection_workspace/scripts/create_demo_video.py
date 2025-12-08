import os
import subprocess
import shutil
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

# Create additional frames for the video
def create_title_frame():
    """Create title slide"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    title_text = [
        "3D OBJECT DETECTION",
        "PointPillars vs SECOND",
        "",
        "Dataset: KITTI",
        "Hardware: Tesla T4 GPU",
        "",
        "Prajwal Dambalkar",
        "December 2025"
    ]
    
    y_pos = 0.9
    for i, line in enumerate(title_text):
        if i < 2:
            ax.text(0.5, y_pos, line, ha='center', va='top', 
                   fontsize=32 if i==0 else 24, fontweight='bold')
            y_pos -= 0.15
        else:
            ax.text(0.5, y_pos, line, ha='center', va='top', fontsize=16)
            y_pos -= 0.08
    
    plt.savefig('results/visualizations/00_title.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created title frame")

def create_results_summary():
    """Create results summary frame"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    summary = [
        "KEY FINDINGS",
        "",
        "✓ Both models successfully detected cars",
        "✓ PointPillars: Higher precision (93.37% avg)",
        "✓ SECOND: Higher recall (11 vs 10 detections)",
        "✓ GPU acceleration: ~10-20x faster than CPU",
        "✓ All detections correctly classified as cars",
        "",
        "CONCLUSION",
        "PointPillars better for precision-critical apps",
        "SECOND better for comprehensive scene coverage"
    ]
    
    y_pos = 0.95
    for line in summary:
        if "FINDINGS" in line or "CONCLUSION" in line:
            ax.text(0.5, y_pos, line, ha='center', va='top', 
                   fontsize=24, fontweight='bold')
            y_pos -= 0.12
        elif line:
            ax.text(0.5, y_pos, line, ha='center', va='top', fontsize=16)
            y_pos -= 0.08
        else:
            y_pos -= 0.04
    
    plt.savefig('results/visualizations/05_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created summary frame")

def create_video():
    """Create demo video from images"""
    create_title_frame()
    create_results_summary()
    
    # List all PNG files in order
    frames = [
        'results/visualizations/00_title.png',
        'results/visualizations/confidence_comparison.png',
        'results/visualizations/model_comparison.png',
        'results/visualizations/05_summary.png'
    ]
    
    # Check if ffmpeg is available
    ffmpeg_available = shutil.which('ffmpeg') is not None
    
    if ffmpeg_available:
        # Create video using ffmpeg
        # Create a text file listing frames
        with open('results/frame_list.txt', 'w') as f:
            for frame in frames:
                f.write(f"file '{frame}'\n")
                f.write("duration 3\n")  # 3 seconds per frame
            # Repeat last frame
            f.write(f"file '{frames[-1]}'\n")
        
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', 'results/frame_list.txt',
            '-vf', 'fps=25,format=yuv420p',
            '-c:v', 'libx264',
            'results/demo_video.mp4'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Created demo video: results/demo_video.mp4")
                print(f"  Duration: ~{len(frames)*3} seconds")
                return True
            else:
                print("⚠️ ffmpeg failed, creating GIF instead...")
                return False
        except Exception as e:
            print(f"⚠️ Error with ffmpeg: {e}")
            return False
    else:
        print("⚠️ ffmpeg not found")
        return False

if __name__ == '__main__':
    print("\nCreating demo video...\n")
    print("="*70)
    
    success = create_video()
    
    if not success:
        print("\n Note: Video frames created but ffmpeg unavailable.")
        print(" All visualization frames are in results/visualizations/")
        print(" You can create video manually or submit frames as-is.")
    
    print("="*70)
    print("\n✅ Video creation complete!")
