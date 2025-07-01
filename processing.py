# processing.py
"""
Multi-volume-alignment-average tool by Taehoon Kim
"""

"""
OCT registration and average
"""

import numpy as np
import cv2
import tifffile as tiff
import SimpleITK as sitk
from scipy.signal import fftconvolve
from tqdm import tqdm
import os
import gc

# 로컬 모듈에서 설정과 유틸리티 함수를 임포트합니다.
from config import CONFIG
from utils import save_volume, save_enface_comparison, save_displacement_plot


# --- Image pre-processing ---

def trim_rows(volume, rows_to_trim):
    return volume[rows_to_trim:, :, :]


def apply_zero_padding(volume, pad_size):
    padded_volume = np.pad(volume, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='constant')
    blank_bscans = np.zeros((padded_volume.shape[0], padded_volume.shape[1], pad_size), dtype=padded_volume.dtype)
    padded_volume = np.concatenate((blank_bscans, padded_volume, blank_bscans), axis=2)
    return padded_volume


def rearrange_to_cscan(volume):
    return np.transpose(volume, (2, 1, 0))


def compute_aip(volume):
    return np.mean(volume, axis=2).astype(np.uint8)


def enhance_contrast(image):
    clip_limit = CONFIG.get('clipLimit', 1.0)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(image)


def fast_std_projection(volume):
    return np.std(volume, axis=2)


def process_volume_for_enface(volume):
    normalized = volume.astype(np.float32) / np.max(volume)
    projection = fast_std_projection(normalized)
    enhanced = enhance_contrast((projection * 255).astype(np.uint8))
    return enhanced


def parallel_preprocess_enfaces(volumes):
    return [process_volume_for_enface(volume) for volume in volumes]


# --- Rigid Registration ---

def rigid_registration_aip_images(aip_images):
    reference_image = aip_images[0]
    registered_images = [reference_image]
    transformations = [sitk.Euler2DTransform()]  # 첫 번째 이미지는 변환 없음

    fixed_image = sitk.GetImageFromArray(reference_image)
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)

    for i in range(1, len(aip_images)):
        moving_image = sitk.GetImageFromArray(aip_images[i])
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=24)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.5)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                          convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        final_transform = registration_method.Execute(fixed_image, moving_image)

        registered_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                         moving_image.GetPixelID())

        registered_images.append(sitk.GetArrayFromImage(registered_image))
        transformations.append(final_transform)

    return registered_images, transformations


def apply_transformation_to_volume(volume, transform):
    transformed_volume = np.zeros_like(volume)
    for z in range(volume.shape[2]):
        slice_image = sitk.GetImageFromArray(volume[:, :, z])
        slice_image = sitk.Cast(slice_image, sitk.sitkFloat32)
        transformed_slice = sitk.Resample(slice_image, slice_image, transform, sitk.sitkLinear, 0.0,
                                          slice_image.GetPixelID())
        transformed_volume[:, :, z] = sitk.GetArrayFromImage(transformed_slice)
    return transformed_volume


# --- En-face and B-scan alignment ---

def phase_correlation(im1, im2):
    f1 = np.fft.fft2(im1)
    f2 = np.fft.fft2(im2)
    cross_power = f1 * np.conj(f2)
    cross_power_norm = cross_power / np.abs(cross_power)
    r = np.fft.ifft2(cross_power_norm)
    y, x = np.unravel_index(np.argmax(np.abs(r)), r.shape)
    if y > im1.shape[0] // 2: y -= im1.shape[0]
    if x > im1.shape[1] // 2: x -= im1.shape[1]
    return x, y


def align_images(im1, im2):
    if im1.shape != im2.shape:
        raise ValueError("Images must have the same dimensions")
    im1_float = im1.astype(float) / 255.0
    im2_float = im2.astype(float) / 255.0

    shift_x, shift_y = phase_correlation(im1_float, im2_float)
    m = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
    height, width = im2.shape
    im1Reg = cv2.warpAffine(im1, m, (width, height))
    return im1Reg, m


def apply_alignment_to_volume(volume, m):
    aligned_volume = np.zeros_like(volume)
    for z in range(volume.shape[2]):
        aligned_volume[:, :, z] = cv2.warpAffine(volume[:, :, z], m, (volume.shape[1], volume.shape[0]))
    return aligned_volume


def detect_non_zero_region(b_scan):
    non_zero_cols = np.where(np.any(b_scan > 0, axis=0))[0]
    if len(non_zero_cols) == 0: return None, None
    start_col, end_col = non_zero_cols[0], non_zero_cols[-1] + 1
    return start_col, end_col


def compute_displacement(b_scan1, b_scan2):
    start_col1, end_col1 = detect_non_zero_region(b_scan1)
    start_col2, end_col2 = detect_non_zero_region(b_scan2)

    if start_col1 is None or start_col2 is None: return 0
    start_col, end_col = max(start_col1, start_col2), min(end_col1, end_col2)
    if start_col >= end_col: return 0

    column_displacements = np.zeros(end_col - start_col)
    for col in range(start_col, end_col):
        cross_corr = fftconvolve(b_scan1[:, col], b_scan2[::-1, col], mode='same')
        max_idx = np.argmax(cross_corr)
        column_displacements[col - start_col] = max_idx - len(b_scan1[:, col]) // 2
    return np.median(column_displacements)


def align_bscans(large_volume):
    num_bscans = large_volume.shape[2]
    aligned_volume = large_volume.copy()
    displacements = np.zeros(num_bscans)

    # tqdm can be used to show progress in the terminal, separately from the GUI.
    for i in tqdm(range(1, num_bscans), desc="Aligning B-scans", leave=False, bar_format='{l_bar}{bar:20}{r_bar}'):
        ref_bscan = aligned_volume[:, :, i - 1].astype(np.float32)
        target_bscan = large_volume[:, :, i].astype(np.float32)
        displacement = compute_displacement(ref_bscan, target_bscan)
        displacements[i] = displacement

        start_col, end_col = detect_non_zero_region(target_bscan)
        if start_col is not None and end_col is not None:
            for j in range(start_col, end_col):
                column = target_bscan[:, j]
                shifted_column = np.interp(np.arange(large_volume.shape[0]) - displacement,
                                           np.arange(large_volume.shape[0]), column,
                                           left=np.nan, right=np.nan)
                shifted_column[np.isnan(shifted_column)] = column[np.isnan(shifted_column)]
                aligned_volume[:, j, i] = shifted_column.astype(np.uint8)
    return aligned_volume, displacements


def rearrange_volumes_for_bscan_alignment(aligned_bscan_volumes):
    num_volumes = len(aligned_bscan_volumes)
    num_bscans = aligned_bscan_volumes[0].shape[2]
    height, width = aligned_bscan_volumes[0].shape[:2]
    large_volume = np.zeros((height, width, num_volumes * num_bscans), dtype=np.uint8)
    for b in range(num_bscans):
        for v in range(num_volumes):
            large_volume[:, :, b * num_volumes + v] = aligned_bscan_volumes[v][:, :, b]
    return large_volume


def average_batch_frames(batch_volume, batch_size):
    num_frames = batch_volume.shape[2]
    averaged_volume = np.zeros((batch_volume.shape[0], batch_volume.shape[1], num_frames // batch_size), dtype=np.uint8)
    for i in range(0, num_frames, batch_size):
        averaged_volume[:, :, i // batch_size] = np.mean(batch_volume[:, :, i:i + batch_size], axis=2).astype(np.uint8)
    return averaged_volume


# --- Main pipeline ---

def load_and_preprocess_volume(filepath):
    """Load a single volume and preprocess it according to the settings"""
    info = tiff.TiffFile(filepath)
    volume_data = np.stack([page.asarray() for page in info.pages], axis=2)

    if volume_data.dtype != np.uint8:
        volume_data = cv2.normalize(volume_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if CONFIG['rows_to_trim'] > 0:
        volume_data = trim_rows(volume_data, CONFIG['rows_to_trim'])
    if CONFIG['pad_size'] > 0:
        volume_data = apply_zero_padding(volume_data, CONFIG['pad_size'])
    return volume_data


def run_alignment_pipeline_on_batch(volumes, save_dir, batch_number, log_callback, progress_callback, p_start, p_range):
    """Execute the complete alignment pipeline for a single batch"""
    steps = 7

    def update(step, msg):
        progress_callback(p_start + (step / steps) * p_range, f"Batch {batch_number}: {msg}")
        log_callback(msg)

    update(0, "Computing AIP images for rigid registration...")
    aip_images = [compute_aip(volume) for volume in volumes]

    update(1, "Performing rigid registration on AIP images...")
    _, transformations = rigid_registration_aip_images(aip_images)

    update(2, "Applying rigid transformations to volumes...")
    aip_aligned_volumes = [volumes[0]]  # 기준 볼륨
    for i in range(1, len(volumes)):
        aligned_volume = apply_transformation_to_volume(volumes[i], transformations[i])
        aip_aligned_volumes.append(aligned_volume)

    update(3, "Performing en face registration...")
    cscan_volumes = [rearrange_to_cscan(volume) for volume in aip_aligned_volumes]
    en_faces = parallel_preprocess_enfaces(cscan_volumes)

    update(4, "Applying en face alignment to volumes...")
    reference_en_face = en_faces[0]
    aligned_cscan_volumes = [cscan_volumes[0]]
    aligned_en_faces = [reference_en_face]

    for i in range(1, len(cscan_volumes)):
        aligned_en_face, h_matrix = align_images(en_faces[i], reference_en_face)
        aligned_volume = apply_alignment_to_volume(cscan_volumes[i], h_matrix)
        aligned_cscan_volumes.append(aligned_volume)
        aligned_en_faces.append(aligned_en_face)

    save_enface_comparison(en_faces, aligned_en_faces, save_dir, batch_number)

    update(5, "Aligning B-scans...")
    bscan_volumes = [np.transpose(vol, (2, 1, 0)) for vol in aligned_cscan_volumes]
    rearranged_for_bscan_alignment = rearrange_volumes_for_bscan_alignment(bscan_volumes)
    aligned_bscan_volume, displacements = align_bscans(rearranged_for_bscan_alignment)

    update(6, "Averaging and saving final volume...")
    averaged_volume = average_batch_frames(aligned_bscan_volume, len(volumes))

    save_path = os.path.join(save_dir, f"aligned_averaged_batch_{batch_number}.tif")
    save_volume(averaged_volume, save_path, log_callback)
    save_displacement_plot(displacements, save_dir, batch_number)

    log_callback(f"Batch {batch_number} completed successfully.")


def process_all_volumes(filepaths, save_dir, log_callback, progress_callback):
    """Main function that processes all selected files in batches"""
    total_volumes = len(filepaths)
    batch_size = CONFIG['batch_size']
    total_batches = (total_volumes + batch_size - 1) // batch_size

    log_callback(f"Processing {total_volumes} volumes in {total_batches} batches of up to {batch_size}.")

    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_volumes)
        current_batch_paths = filepaths[start_idx:end_idx]
        batch_number = i + 1

        # Calculate overall progress
        batch_progress_start = 10 + (i / total_batches) * 80
        batch_progress_range = 80 / total_batches

        progress_callback(batch_progress_start, f"Processing batch {batch_number}/{total_batches}")
        log_callback(f"--- Starting Batch {batch_number}: files {start_idx + 1} to {end_idx} ---")

        # Load volume
        volumes = []
        for j, filepath in enumerate(current_batch_paths):
            p_load = batch_progress_start + (j / len(current_batch_paths)) * (batch_progress_range * 0.1)  # 로딩은 10% 차지
            progress_callback(p_load, f"Loading {os.path.basename(filepath)}")
            log_callback(f"Loading: {os.path.basename(filepath)}")
            volumes.append(load_and_preprocess_volume(filepath))

        # Execute the processing pipeline
        p_pipeline_start = batch_progress_start + (batch_progress_range * 0.1)
        p_pipeline_range = batch_progress_range * 0.9
        run_alignment_pipeline_on_batch(volumes, save_dir, batch_number, log_callback, progress_callback,
                                        p_pipeline_start, p_pipeline_range)

        del volumes
        gc.collect()

    progress_callback(95, "Finalizing...")