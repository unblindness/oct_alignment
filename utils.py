# utils.py
"""
Multi-volume-alignment-average tool by Taehoon Kim
"""

"""
File saving, visualization
"""

import os
import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np


def save_volume(volume, filepath, log_callback=None):
    """Save the OCT volume as a TIFF file.."""
    if log_callback:
        log_callback(f'Saving aligned volume to {os.path.basename(filepath)}')
    with tiff.TiffWriter(filepath) as t:
        for z in range(volume.shape[2]):
            t.write(volume[:, :, z], photometric='minisblack')


def save_enface_comparison(original_images, aligned_images, save_dir, batch_number):
    """Save images comparing the state before and after en-face alignment."""
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    avg_original = np.mean(original_images, axis=0).astype(np.uint8)
    avg_aligned = np.mean(aligned_images, axis=0).astype(np.uint8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(avg_original, cmap='gray')
    ax1.set_title('Average Original')
    ax1.axis('off')
    ax2.imshow(avg_aligned, cmap='gray')
    ax2.set_title('Average Aligned')
    ax2.axis('off')

    plt.tight_layout()
    save_path = os.path.join(vis_dir, f'enface_comparison_batch_{batch_number}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def save_displacement_plot(displacements, save_dir, batch_number):
    """Save the B-scan displacement graph."""
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(displacements)
    plt.title(f'B-scan Displacements (Batch {batch_number})')
    plt.xlabel('B-scan Index')
    plt.ylabel('Displacement')

    displacement_path = os.path.join(vis_dir, f'displacement_batch_{batch_number}.png')
    plt.savefig(displacement_path, dpi=300, bbox_inches='tight')
    plt.close()