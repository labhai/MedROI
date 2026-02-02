import os
import glob
import numpy as np
import nibabel as nib
import cv2
import pandas as pd
from scipy import ndimage
import time
import json
import logging

# Import pydicom
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not installed. DICOM support disabled.")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parameters
SUBJECTS_ROOT = "/path/to/dicom/files"
NIFTI_PATTERN = "*.nii"
OUT_DIR = "./bbox_outputs"
SAVE_CROPPED = True

MISS_THR = 0.002
PADDING = 3
EXTRA_PAD = 6

# Utility functions
def bbox_from_mask(mask_bool):
    if not np.any(mask_bool): return None
    ys, xs = np.where(mask_bool)
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())

def pad_box(box, H, W, pad):
    if box is None: return None
    y0, x0, y1, x1 = box
    return max(0, y0-pad), max(0, x0-pad), min(H-1, y1+pad), min(W-1, x1+pad)

def normalize_to_uint8(V):
    """Normalize volume to uint8 range considering original intensity distribution"""
    V_nz = V[V > 0]
    if V_nz.size == 0:
        return np.zeros_like(V, dtype=np.uint8)
    
    vmin = np.percentile(V_nz, 1)
    vmax = np.percentile(V_nz, 99)
    
    V_normalized = np.clip((V - vmin) / (vmax - vmin) * 255, 0, 255)
    return V_normalized.astype(np.uint8)

# Debugging function
def debug_bbox_computation(img_2d, subject_id):
    """Debug bounding box computation process"""
    img_uint8 = normalize_to_uint8(img_2d)
    H, W = img_uint8.shape
    nz = img_uint8 > 0
    
    print(f"\n{'='*60}")
    print(f"Debugging: {subject_id}")
    print(f"{'='*60}")
    
    # Original statistics
    print(f"\nOriginal Image Stats:")
    print(f"  Shape: {img_2d.shape}")
    print(f"  Value range: [{img_2d.min():.2f}, {img_2d.max():.2f}]")
    print(f"  Non-zero pixels: {nz.sum():,} ({nz.sum()/img_2d.size*100:.2f}%)")
    
    # Normalized statistics
    print(f"\nNormalized (uint8) Stats:")
    print(f"  Value range: [{img_uint8.min()}, {img_uint8.max()}]")
    print(f"  Mean (non-zero): {img_uint8[nz].mean():.2f}")
    
    # Threshold calculation
    t_global = float(np.percentile(img_uint8[nz], 25))
    print(f"\nThreshold Values:")
    print(f"  Global threshold: {t_global:.2f}")
    
    # Reference mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ref_bool = (img_uint8 > 0)
    ref_pixels = int(ref_bool.sum())  
    
    print(f"\nReference Mask:")
    print(f"  Pixels in mask: {ref_pixels:,} ({ref_pixels/img_2d.size*100:.2f}%)")
    
    # Bbox calculation
    non_zero_pixels = img_uint8[img_uint8 > 0]
    thr_slice = float(np.percentile(non_zero_pixels, 10))
    print(f"  Slice threshold: {thr_slice:.2f}")
    
    box_a = bbox_from_mask(img_uint8 > thr_slice)
    if box_a:
        box_a = pad_box(box_a, H, W, PADDING)
        ya, xa, yb, xb = box_a
        
        bbox_area = (yb - ya + 1) * (xb - xa + 1)
        print(f"\nBbox (with padding={PADDING}):")
        print(f"  Coordinates: ({ya}, {xa}) -> ({yb}, {xb})")
        print(f"  Size: {yb-ya+1} x {xb-xa+1} = {bbox_area:,} pixels")
        
        # Coverage check
        nz_mask = (img_uint8 > 0)
        nz_pixels = nz_mask.sum()

        bbA = np.zeros_like(nz_mask, bool)
        bbA[ya:yb+1, xa:xb+1] = True

        covered = (nz_mask & bbA).sum()
        missed = (nz_mask & (~bbA)).sum()
        miss_rate = missed / nz_pixels if nz_pixels > 0 else 0
        
        print(f"\nCoverage Analysis:")
        print(f"  Covered: {covered:,} ({covered/ref_pixels*100:.2f}%)")
        print(f"  Missed: {missed:,} ({miss_rate*100:.4f}%)")
        print(f"  Miss threshold: {MISS_THR*100:.4f}%")
        print(f"  Status: {'NEEDS EXTRA PADDING' if miss_rate > MISS_THR else 'OK'}")
        
        # Apply extra padding if needed
        final_box = box_a
        if miss_rate > MISS_THR:
            final_box = pad_box(box_a, H, W, PADDING + EXTRA_PAD)
            if final_box:
                print(f"  -> Extra padded: ({final_box[0]}, {final_box[1]}) -> ({final_box[2]}, {final_box[3]})")
        
        print(f"\nVisualization:")
        print(f"  Original center: ({H//2}, {W//2})")
        print(f"  Bbox center: ({(ya+yb)//2}, {(xa+xb)//2})")
        print(f"  Offset: ({(ya+yb)//2 - H//2}, {(xa+xb)//2 - W//2})")
        
        print(f"{'='*60}\n")
        return final_box
    else:
        print(f"\nNo bbox found!")
        print(f"{'='*60}\n")
        return None

# Bbox computation function
def compute_2d_bbox(img_2d):
    """Compute bounding box from 2D image"""
    img_uint8 = normalize_to_uint8(img_2d)
    
    H, W = img_uint8.shape
    nz = img_uint8 > 0
    
    if not np.any(nz):
        return None
    
    t_global = float(img_uint8[nz].mean())
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    ref_bool = (img_uint8 > 0) 
    ref_pixels = int(ref_bool.sum())
    
    if ref_pixels == 0:
        return None
    
    non_zero_pixels = img_uint8[img_uint8 > 0]
    thr_slice = float(np.percentile(non_zero_pixels, 10))
    
    box_a = bbox_from_mask(img_uint8 > thr_slice)
    box_a = pad_box(box_a, H, W, PADDING)
    
    if box_a is None:
        return None
    
    ya, xa, yb, xb = box_a
    bbA = np.zeros_like(ref_bool, bool)
    bbA[ya:yb+1, xa:xb+1] = True
    miss_a = (ref_bool & (~bbA)).sum() / ref_pixels
    
    final_box = box_a
    if miss_a > MISS_THR:
        final_box = pad_box(box_a, H, W, PADDING + EXTRA_PAD)
    
    return final_box

# DICOM processing function
def process_single_2d_dicom(dcm_path, output_dir, enable_debug=False):
    """Process single 2D DICOM file"""
    if not PYDICOM_AVAILABLE:
        raise RuntimeError("pydicom is required for DICOM processing")
    
    start_time = time.time()
    basename = os.path.basename(dcm_path).replace('.dcm', '')
    
    logger.info(f"Processing 2D DICOM: {basename}")
    
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.float32)
    
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        img = img * ds.RescaleSlope + ds.RescaleIntercept
    
    original_shape = img.shape
    logger.info(f"  Original shape: {original_shape}")
    logger.info(f"  Value range: [{img.min():.2f}, {img.max():.2f}]")
    
    bbox_start = time.time()
    
    # Debug mode
    if enable_debug:
        bbox = debug_bbox_computation(img, basename)
    else:
        bbox = compute_2d_bbox(img)
    
    bbox_time = time.time() - bbox_start
    
    if bbox is None:
        logger.warning(f"  No valid bbox found")
        return None
    
    y0, x0, y1, x1 = bbox
    logger.info(f"  Bbox: ({y0}, {x0}) -> ({y1}, {x1})")
    
    cropped = img[y0:y1+1, x0:x1+1]
    cropped_shape = cropped.shape
    logger.info(f"  Cropped shape: {cropped_shape}")
    
    # Save NPY file
    save_start = time.time()
    npy_path = os.path.join(output_dir, "cropped_2d_npy", f"{basename}_cropped.npy")
    np.save(npy_path, cropped)
    save_time = time.time() - save_start
    
    npy_size = os.path.getsize(npy_path)
    logger.info(f"  Saved to: {npy_path} ({npy_size:,} bytes)")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "dicom_metadata", f"{basename}_metadata.json")
    metadata_dict = {
        'bbox': {'y_min': int(y0), 'x_min': int(x0), 'y_max': int(y1), 'x_max': int(x1)},
        'original_shape': list(original_shape),
        'cropped_shape': list(cropped_shape),
        'original_file': dcm_path,
        'npy_file': npy_path,
        'value_range': {
            'min': float(img.min()), 'max': float(img.max()),
            'cropped_min': float(cropped.min()), 'cropped_max': float(cropped.max())
        },
        'modality': str(getattr(ds, 'Modality', 'Unknown')),
        'bits_stored': int(getattr(ds, 'BitsStored', 0))
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    
    total_time = time.time() - start_time
    
    result = {
        'filename': basename,
        'original_file': dcm_path,
        'npy_file': npy_path,
        'metadata_file': metadata_path,
        'original_shape': f"{original_shape[0]}x{original_shape[1]}",
        'cropped_shape': f"{cropped_shape[0]}x{cropped_shape[1]}",
        'bbox': f"({y0},{x0})-({y1},{x1})",
        'npy_size_bytes': npy_size,
        'bbox_calc_time': bbox_time,
        'file_save_time': save_time,
        'total_time': total_time,
        'modality': str(getattr(ds, 'Modality', 'Unknown')),
        'bits_stored': int(getattr(ds, 'BitsStored', 0))
    }
    
    logger.info(f"  Completed in {total_time:.4f}s")
    
    return result

# DICOM file verification functions
def is_dicom_file(filepath):
    if not PYDICOM_AVAILABLE:
        return False
    try:
        pydicom.dcmread(filepath, stop_before_pixels=True)
        return True
    except:
        return False

def is_dicom_folder(path):
    if not PYDICOM_AVAILABLE or not os.path.isdir(path):
        return False
    
    files = os.listdir(path)
    for file in files[:10]:
        filepath = os.path.join(path, file)
        if os.path.isfile(filepath) and is_dicom_file(filepath):
            return True
    return False

# Main function
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if SAVE_CROPPED:
        os.makedirs(os.path.join(OUT_DIR, "cropped_2d_npy"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "dicom_metadata"), exist_ok=True)
    
    # Debug mode settings
    ENABLE_DEBUG = True  # Set to True for detailed output on all files
    
    # For debugging specific files only:
    # DEBUG_FILES = ['I123456.dcm', 'I789012.dcm']  # Problematic filenames
    DEBUG_FILES = []  # Leave empty for full debugging
    
    # Check DICOM folder
    if not is_dicom_folder(SUBJECTS_ROOT):
        logger.error(f"No valid DICOM files found in {SUBJECTS_ROOT}")
        return
    
    # Collect DICOM files
    dicom_files = []
    for root, dirs, files in os.walk(SUBJECTS_ROOT):
        for file in files:
            filepath = os.path.join(root, file)
            if is_dicom_file(filepath):
                dicom_files.append(filepath)
    
    dicom_files.sort()
    
    print(f"\n{'='*60}")
    print(f"Processing {len(dicom_files)} 2D DICOM files")
    print(f"Debugging mode: {'ALL FILES' if ENABLE_DEBUG and not DEBUG_FILES else 'SPECIFIC FILES' if DEBUG_FILES else 'OFF'}")
    print(f"{'='*60}\n")
    
    results = []
    success_count = 0
    
    for dcm_path in dicom_files:
        try:
            basename = os.path.basename(dcm_path)
            
            # Set debugging condition
            if DEBUG_FILES:
                enable_debug = basename in DEBUG_FILES
            else:
                enable_debug = ENABLE_DEBUG
            
            result = process_single_2d_dicom(dcm_path, OUT_DIR, enable_debug=enable_debug)
            if result:
                results.append(result)
                success_count += 1
        except Exception as e:
            logger.error(f"Error processing {os.path.basename(dcm_path)}: {e}")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(OUT_DIR, "cropped_2d_dicom_results.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"Processing Statistics:")
        print(f"  Total files: {len(dicom_files)}")
        print(f"  Successfully processed: {success_count}")
        print(f"  Average total time: {df['total_time'].mean():.4f}s")
        print(f"  Total .npy size: {df['npy_size_bytes'].sum() / 1024 / 1024:.2f} MB")
        print(f"\nResults saved to: {csv_path}")
        print(f"Cropped .npy files: {os.path.join(OUT_DIR, 'cropped_2d_npy')}")
        print(f"Metadata files: {os.path.join(OUT_DIR, 'dicom_metadata')}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()