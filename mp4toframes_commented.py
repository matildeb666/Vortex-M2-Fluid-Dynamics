#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 11:19:02 2026

@author: matilde bureau ,gaston ravanas.

This script takes a .mp4 basilisk animation and extracts a certain number of frames
from it to be usable for a written report. 



"""

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import re
import glob
import shutil

# ==========================================
# 1. Parameter & Case Detection
# ==========================================

def extract_param_value(filename, param_type):
    """Uses regex to extract physical constants from the filename for plot labeling."""
    pattern = rf"{param_type}_([0-9]+\.?[0-9]*)"
    match = re.search(pattern, filename, re.IGNORECASE)
    return match.group(1) if match else None

def detect_case(filename):
    """Categorizes the simulation to apply specific frame-sampling strategies."""
    fname_lower = os.path.basename(filename).lower()
    for case in ['corot', 'dipolar', 'dipole', 'single', 'image']:
        if case in fname_lower:
            return case
    return 'other'

# ==========================================
# 2. Core Video Processing Logic
# ==========================================

def process_vortex_video(video_path):
    """
    Extracts key frames from a simulation video and organizes them into a PDF grid.
    Includes custom sampling rates based on the physics of the case.
    """
    basename = os.path.splitext(os.path.basename(video_path))[0]
    case_folder = os.path.dirname(video_path)
    case_type = detect_case(basename)

    # Compile parameters for the output filename
    params = {
        'Re': extract_param_value(basename, 'Re'),
        'G2': extract_param_value(basename, 'G2'),
        'a2': extract_param_value(basename, 'a2')
    }
    param_suffix = "_".join([f"{k}_{v}" for k, v in params.items() if v is not None])
    out_name = f"{case_folder}_{param_suffix}_frames.pdf" if param_suffix else f"{case_folder}_frames.pdf"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- Frame Selection Logic (Heuristic based on Physics) ---
    # We use biased sampling because vortex interactions often have 
    # high-frequency changes early on and slow down later.
    
    if case_type == 'image':
        # Samples 8 frames from early interaction (up to 19s) and 4 frames from late stage
        print(f"  -> Case 'image': Applying 12-frame biased sampling.")
        early_indices = np.linspace(0, min(19 * fps, total_frames - 1), 8, dtype=int)
        late_indices = np.linspace(min(19 * fps + 1, total_frames - 1), total_frames - 1, 4, dtype=int)
        frame_indices = np.concatenate([early_indices, late_indices])
        grid_size, figsize = (4, 3), (15, 18)

    elif case_type in ['dipole', 'dipolar']:
        # Dipoles typically move linearly; even sampling is sufficient
        print(f"  -> Case '{case_type}': Applying 12-frame EVEN sampling.")
        frame_indices = np.linspace(0, total_frames - 1, 12, dtype=int)
        grid_size, figsize = (4, 3), (15, 18)

    elif case_type == 'corot':
        # Corotating vortices interact rapidly in the first 4 seconds
        print(f"  -> Case 'corot': Biased sampling (split at 4s).")
        early_indices = np.linspace(0, min(4 * fps, total_frames - 1), 8, dtype=int)
        late_indices = np.linspace(min(4 * fps + 1, total_frames - 1), total_frames - 1, 4, dtype=int)
        frame_indices = np.unique(np.concatenate([early_indices, late_indices]))
        grid_size, figsize = (4, 3), (15, 18)

    else:
        # Default 6-frame grid for simple/unknown cases
        print(f"  -> Case '{case_type}': Default 6-frame sampling.")
        frame_indices = np.linspace(0, total_frames - 1, 6, dtype=int)
        grid_size, figsize = (2, 3), (15, 10)
    
    # --- Extraction and Time-Stamping ---
    extracted_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success: break

        # Burn the simulation time into the frame for clarity
        timestamp = idx / fps
        time_text = f"t = {timestamp:.3f} s"
        cv2.putText(frame, time_text, (30, 50), cv2.FONT_HERSHEY_DUPLEX, 
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        extracted_frames.append(frame_rgb)

    cap.release()

    # --- Visualization Construction ---
    if not extracted_frames: return

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    axes = axes.flatten()
    
    for j, img in enumerate(extracted_frames):
        if j < len(axes):
            axes[j].imshow(img)
            axes[j].axis('off')

    # Cleanup: remove axes from unused grid slots
    for k in range(len(extracted_frames), len(axes)):
        axes[k].axis('off')

    plt.tight_layout()
    save_path = os.path.join(case_folder, out_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"      [Success] Saved grid to {save_path}")

# ==========================================
# 3. Batch Execution
# ==========================================

if __name__ == "__main__":
    video_files = glob.glob("*.mp4")
    if video_files:
        print(f"Batch Processing Started: Found {len(video_files)} video(s).")
        for f in video_files:
            case = detect_case(f)
            os.makedirs(case, exist_ok=True)
            
            # Move file to its case folder and process
            new_path = os.path.join(case, f)
            shutil.move(f, new_path)
            process_vortex_video(new_path)
        print("\nVisualization pipeline complete.")
    else:
        print("No .mp4 files found in the directory.")