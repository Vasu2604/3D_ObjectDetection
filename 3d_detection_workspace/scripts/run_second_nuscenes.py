#!/usr/bin/env python3
"""
SECOND on nuScenes Inference Script
Adds the missing model-dataset combination to complete the assignment
"""
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    from mmdet3d.apis import init_model, inference_detector
    from mmdet3d.registry import VISUALIZERS
    import mmcv
except ImportError as e:
    print(f"Error importing mmdet3d: {e}")
    print("Make sure mmdet3d is installed")
    sys.exit(1)

def main():
    print("=" * 70)
    print("  SECOND on nuScenes - New Model-Dataset Combination")
    print("=" * 70)
    print()
    
    # Paths
    workspace_dir = Path("/teamspace/studios/this_studio/3d_detection_workspace")
    checkpoint_dir = workspace_dir / "checkpoints"
    data_dir = workspace_dir / "data" / "nuscenes"
    results_dir = workspace_dir / "results" / "nuscenes" / "second"
    
    # Create directories
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "images").mkdir(exist_ok=True)
    (results_dir / "pointclouds").mkdir(exist_ok=True)
    (results_dir / "metadata").mkdir(exist_ok=True)
    
    print(f"📁 Workspace: {workspace_dir}")
    print(f"📁 Results will be saved to: {results_dir}")
    print()
    
    # Download checkpoint if needed
    checkpoint_url = "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/second/hv_second_secfpn_6x8_80e_kitti-3d-3class/hv_second_secfpn_6x8_80e_kitti-3d-3class_20210831_022017-ae782e87.pth"
    checkpoint_path = checkpoint_dir / "second_kitti.pth"
    config_file = "/opt/conda/lib/python3.10/site-packages/mmdet3d/configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py"
    
    # Since we're using KITTI checkpoint on nuScenes (transfer learning)
    if not checkpoint_path.exists():
        print(f"📥 Downloading SECOND checkpoint...")
        os.system(f"wget -q {checkpoint_url} -O {checkpoint_path}")
        print(f"✅ Downloaded to {checkpoint_path}")
    else:
        print(f"✅ Using existing checkpoint: {checkpoint_path}")
    
    print()
    
    # Find nuScenes sample file
    sample_files = list(data_dir.glob("**/*.bin"))
    if not sample_files:
        print("❌ No .bin files found in nuScenes data directory")
        print(f"Searched in: {data_dir}")
        return
    
    pcd_file = sample_files[0]
    print(f"📄 Input file: {pcd_file.name}")
    print(f"📍 Full path: {pcd_file}")
    print()
    
    # Initialize model
    print("🧠 Initializing SECOND model...")
    try:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"💻 Device: {device}")
        
        model = init_model(config_file, str(checkpoint_path), device=device)
        print("✅ Model initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return
    
    print()
    
    # Run inference
    print("🚀 Running inference...")
    start_time = time.time()
    
    try:
        result, data = inference_detector(model, str(pcd_file))
        inference_time = time.time() - start_time
        
        print(f"✅ Inference completed in {inference_time:.3f}s")
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return
    
    print()
    
    # Extract results
    pred_bboxes = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
    pred_labels = result.pred_instances_3d.labels_3d.cpu().numpy()
    pred_scores = result.pred_instances_3d.scores_3d.cpu().numpy()
    
    num_detections = len(pred_bboxes)
    
    print("📊 Detection Results:")
    print(f"  Total detections: {num_detections}")
    
    if num_detections > 0:
        print(f"  Score range: [{pred_scores.min():.3f}, {pred_scores.max():.3f}]")
        print(f"  Mean score: {pred_scores.mean():.3f}")
        print(f"  Median score: {np.median(pred_scores):.3f}")
        
        # High confidence detections
        high_conf = (pred_scores >= 0.7).sum()
        print(f"  High confidence (≥0.7): {high_conf}")
        
        # Top 5 detections
        top5_idx = np.argsort(pred_scores)[-5:][::-1]
        print("\n  Top 5 detections:")
        for i, idx in enumerate(top5_idx, 1):
            print(f"    {i}. Score: {pred_scores[idx]:.4f}, Label: {pred_labels[idx]}")
    
    print()
    
    # Save metadata
    metadata = {
        "model": "SECOND",
        "dataset": "nuScenes",
        "checkpoint": str(checkpoint_path.name),
        "config": config_file,
        "input_file": str(pcd_file.name),
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "inference_time_seconds": float(inference_time),
        "num_detections": int(num_detections),
        "scores": {
            "min": float(pred_scores.min()) if num_detections > 0 else 0,
            "max": float(pred_scores.max()) if num_detections > 0 else 0,
            "mean": float(pred_scores.mean()) if num_detections > 0 else 0,
            "median": float(np.median(pred_scores)) if num_detections > 0 else 0,
            "std": float(pred_scores.std()) if num_detections > 0 else 0,
            "high_conf_count": int((pred_scores >= 0.7).sum()) if num_detections > 0 else 0
        },
        "detections": []
    }
    
    # Add detection details
    for i in range(num_detections):
        metadata["detections"].append({
            "id": int(i),
            "score": float(pred_scores[i]),
            "label": int(pred_labels[i]),
            "bbox": pred_bboxes[i].tolist()
        })
    
    # Save JSON
    json_path = results_dir / "metadata" / "detection_results.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata to: {json_path}")
    
    # Save simple summary
    summary_path = results_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("SECOND on nuScenes - Detection Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: SECOND (Sparsely Embedded Convolutional Detection)\n")
        f.write(f"Dataset: nuScenes\n")
        f.write(f"Input: {pcd_file.name}\n")
        f.write(f"Device: {device}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Total Detections: {num_detections}\n")
        if num_detections > 0:
            f.write(f"  Mean Confidence: {pred_scores.mean():.4f}\n")
            f.write(f"  Max Confidence: {pred_scores.max():.4f}\n")
            f.write(f"  High Confidence (≥0.7): {(pred_scores >= 0.7).sum()}\n")
            f.write(f"  Inference Time: {inference_time:.3f}s\n")
    
    print(f"💾 Saved summary to: {summary_path}")
    print()
    
    print("✅ SECOND on nuScenes inference completed successfully!")
    print("=" * 70)
    print()
    print("🎉 New model-dataset combination added to your assignment!")
    print()

if __name__ == "__main__":
    main()
