import os
import sys
import numpy as np
import nibabel as nib
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import json
import argparse
from pathlib import Path
import logging
import time
import pandas as pd
import glob
import io
import cv2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== JPEG2000 Support Check ==========

def check_jpeg2000_support():
    """Check JPEG2000 support availability"""
    try:
        test_img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))
        buffer = io.BytesIO()
        test_img.save(buffer, format='JPEG2000', quality_mode='rates', quality_layers=[20])
        buffer.seek(0)
        Image.open(buffer)
        logger.info("JPEG2000 support confirmed")
        return True
    except Exception as e:
        logger.error(f"JPEG2000 not available: {e}")
        return False

if not check_jpeg2000_support():
    sys.exit(1)


# ========== ROI Metadata Management ==========

class ROIMetadata:
    """Metadata class for ROI-based compression"""
    
    BBOX_SIZE = 12      # 6 values × int16
    SHAPE_SIZE = 6      # 3 values × int16
    AFFINE_SIZE = 36    # 9 values (3×3) × float32
    TOTAL_SIZE = BBOX_SIZE + SHAPE_SIZE + AFFINE_SIZE  # 54 bytes
    
    def __init__(self, bbox=None, original_shape=None, affine_matrix=None):
        self.bbox = bbox
        self.original_shape = original_shape
        self.affine_matrix = affine_matrix
    
    def to_bytes(self):
        """Serialize metadata to bytes"""
        data = b''
        
        # Bounding Box (12 bytes)
        if self.bbox:
            bbox_array = np.array([
                self.bbox['x_min'], self.bbox['x_max'],
                self.bbox['y_min'], self.bbox['y_max'],
                self.bbox['z_min'], self.bbox['z_max']
            ], dtype=np.int16)
            data += bbox_array.tobytes()
        else:
            data += np.zeros(6, dtype=np.int16).tobytes()
        
        # Original Shape (6 bytes)
        if self.original_shape:
            shape_array = np.array(self.original_shape, dtype=np.int16)
            data += shape_array.tobytes()
        else:
            data += np.zeros(3, dtype=np.int16).tobytes()
        
        # Affine Matrix (36 bytes, 3×3 only)
        if self.affine_matrix is not None:
            if self.affine_matrix.shape == (4, 4):
                affine_3x3 = self.affine_matrix[:3, :3].astype(np.float32)
            elif self.affine_matrix.shape == (3, 3):
                affine_3x3 = self.affine_matrix.astype(np.float32)
            else:
                affine_3x3 = np.eye(3, dtype=np.float32)
            data += affine_3x3.tobytes()
        else:
            data += np.eye(3, dtype=np.float32).tobytes()
        
        assert len(data) == self.TOTAL_SIZE
        return data
    
    @classmethod
    def from_bytes(cls, data):
        """Deserialize metadata from bytes"""
        assert len(data) == cls.TOTAL_SIZE
        
        offset = 0
        
        # Bounding Box
        bbox_array = np.frombuffer(data[offset:offset+cls.BBOX_SIZE], dtype=np.int16)
        bbox = {
            'x_min': int(bbox_array[0]), 'x_max': int(bbox_array[1]),
            'y_min': int(bbox_array[2]), 'y_max': int(bbox_array[3]),
            'z_min': int(bbox_array[4]), 'z_max': int(bbox_array[5])
        }
        offset += cls.BBOX_SIZE
        
        # Original Shape
        shape_array = np.frombuffer(data[offset:offset+cls.SHAPE_SIZE], dtype=np.int16)
        original_shape = tuple(int(x) for x in shape_array)
        offset += cls.SHAPE_SIZE
        
        # Affine Matrix
        affine_3x3 = np.frombuffer(data[offset:offset+cls.AFFINE_SIZE], 
                                   dtype=np.float32).reshape(3, 3)
        affine_4x4 = np.eye(4, dtype=np.float32)
        affine_4x4[:3, :3] = affine_3x3
        
        return cls(bbox=bbox, original_shape=original_shape, affine_matrix=affine_4x4)


# ========== Image Processing ==========

class BrainImageProcessor:
    """Brain image preprocessing and normalization"""
    
    def __init__(self, normalize_method='minmax'):
        self.normalize_method = normalize_method
        
    def load_nifti(self, nii_path):
        """Load NIfTI file with header and affine"""
        try:
            img = nib.load(nii_path)
            data = img.get_fdata()
            return data, img.header, img.affine
        except Exception as e:
            logger.error(f"Error loading {nii_path}: {e}")
            return None, None, None
    
    def normalize_image(self, image_data):
        """Normalize image to [0, 255] range"""
        if self.normalize_method == 'minmax':
            min_val, max_val = np.min(image_data), np.max(image_data)
            if max_val > min_val:
                return ((image_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        elif self.normalize_method == 'zscore':
            mean_val, std_val = np.mean(image_data), np.std(image_data)
            if std_val > 0:
                z_scored = (image_data - mean_val) / std_val
                return np.clip((z_scored + 3) / 6 * 255, 0, 255).astype(np.uint8)
        
        return np.zeros_like(image_data, dtype=np.uint8)
    
    def extract_slices(self, volume_data, axis=2, max_slices=None):
        """
        Extract 2D slices from 3D volume.
        
        Args:
            volume_data: 3D volume array
            axis: slice axis (0=sagittal, 1=coronal, 2=axial)
            max_slices: maximum number of slices to extract
            
        Returns:
            slices: list of normalized 2D slices
        """
        slice_count = volume_data.shape[axis]
        
        if max_slices is None or max_slices >= slice_count:
            indices = range(slice_count)
        else:
            step = max(1, slice_count // max_slices)
            indices = range(slice_count // 4, 3 * slice_count // 4, step)[:max_slices]
        
        slices = []
        for i in indices:
            slice_data = np.take(volume_data, i, axis=axis)
            slices.append(self.normalize_image(slice_data))
        
        return slices


# ========== JPEG2000 Compression ==========

class JPEG2000Compressor:
    """JPEG2000 compressor with quality levels"""
    
    COMPRESSION_RATES = {
        1: [100],  # Highest quality
        2: [50],   # High quality
        3: [20],   # Medium quality
        4: [10],   # Low quality
        5: [5]     # Lowest quality (highest compression)
    }
    
    def compress_image(self, image_array, quality_level=3):
        """
        Compress image using JPEG2000.
        
        Args:
            image_array: 2D numpy array (uint8)
            quality_level: 1-5 (1=best quality, 5=best compression)
            
        Returns:
            compressed_data: dict with compressed data
            comp_time: compression time
        """
        start_time = time.time()
        
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        
        rates = self.COMPRESSION_RATES.get(quality_level, [20])
        
        try:
            pil_img = Image.fromarray(image_array, mode='L')
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG2000', 
                        quality_mode='rates', 
                        quality_layers=rates)
            
            compressed_data = buffer.getvalue()
            comp_time = time.time() - start_time
            
            return {
                'compressed_data': compressed_data,
                'original_shape': image_array.shape,
                'compressed_size': len(compressed_data),
                'method': f'jpeg2000_rate_{rates[0]}'
            }, comp_time
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise
    
    def decompress_image(self, compressed_data):
        """
        Decompress JPEG2000 image.
        
        Args:
            compressed_data: list containing compressed data dict
            
        Returns:
            reconstructed: decompressed image
            decomp_time: decompression time
        """
        start_time = time.time()
        
        try:
            buffer = io.BytesIO(compressed_data[0]['compressed_data'])
            pil_img = Image.open(buffer)
            reconstructed = np.array(pil_img, dtype=np.uint8)
            
            decomp_time = time.time() - start_time
            return reconstructed, decomp_time
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise


# ========== Quality Evaluation ==========

class CompressionEvaluator:
    """Compression quality evaluator"""
    
    @staticmethod
    def calculate_metrics(original, reconstructed):
        """Calculate image quality metrics"""
        mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
        
        if mse == 0:
            psnr = 100.0
        elif mse < 1e-10:
            psnr = 20 * np.log10(255 / np.sqrt(mse))
        else:
            psnr = 20 * np.log10(255 / np.sqrt(mse))
        
        ssim_val = ssim(original, reconstructed, data_range=255)
        
        return mse, psnr, ssim_val


# ========== Utility Functions ==========

def pad_roi_slice_to_full(patch, roi_metadata, original_full_volume=None):
    """
    Pad ROI slice to original full size.
    
    Args:
        patch: reconstructed ROI slice (2D array)
        roi_metadata: ROIMetadata object
        original_full_volume: actual loaded original volume (priority)
        
    Returns:
        canvas: full-size padded slice
    """
    assert patch.ndim == 2
    assert roi_metadata is not None and roi_metadata.bbox is not None

    bbox = roi_metadata.bbox
    x0 = bbox['x_min']
    x1 = bbox['x_max']
    y0 = bbox['y_min']
    y1 = bbox['y_max']

    # Use actual loaded volume shape if available
    if original_full_volume is not None:
        X, Y, Z = original_full_volume.shape
    else:
        assert roi_metadata.original_shape is not None
        X, Y, Z = map(int, roi_metadata.original_shape)

    canvas = np.zeros((X, Y), dtype=patch.dtype)

    expected_x = x1 - x0 + 1
    expected_y = y1 - y0 + 1

    # Shape verification and correction
    if patch.shape != (expected_x, expected_y):
        patch = cv2.resize(
            patch, 
            (expected_y, expected_x),
            interpolation=cv2.INTER_NEAREST
        )
    
    canvas[x0:x1+1, y0:y1+1] = patch
    
    return canvas


def get_target_slice_index(volume_shape, roi_metadata=None):
    """
    Get target slice index for same anatomical position.
    
    Args:
        volume_shape: (H, W, D) current volume shape
        roi_metadata: ROIMetadata object (for ROI)
        
        
    Returns:
        slice_idx: slice index in current volume
        anatomical_position: anatomical position in original volume
    """
    depth = volume_shape[2]
    
    if roi_metadata is None or roi_metadata.bbox is None:
        # Original data
        slice_idx = depth // 2
        anatomical_position = slice_idx
        
    else:
        # ROI data
        z_min = roi_metadata.bbox['z_min']
        z_max = roi_metadata.bbox['z_max']
        
        slice_idx = (z_max - z_min) // 2
        anatomical_position = z_min + slice_idx
    
    return slice_idx, anatomical_position


def load_bbox_info(bbox_csv_path):
    """Load 3D bounding box information from CSV"""
    if not os.path.exists(bbox_csv_path):
        logger.warning(f"BBox file not found: {bbox_csv_path}")
        return {}
    
    try:
        df = pd.read_csv(bbox_csv_path)
        bbox_dict = {}
        
        for _, row in df.iterrows():
            subject_id = row['subject_id']
            
            bbox_dict[subject_id] = {
                'x_min': int(row['x_min']),
                'x_max': int(row['x_max']),
                'y_min': int(row['y_min']),
                'y_max': int(row['y_max']),
                'z_min': int(row['z_min']),
                'z_max': int(row['z_max']),
                'original_shape': (int(row['orig_x']), int(row['orig_y']), int(row['orig_z']))
            }
        
        return bbox_dict
        
    except Exception as e:
        logger.error(f"Failed to load BBox: {e}")
        return {}


def find_file_pairs(cropped_dir, original_dir):
    """Match cropped files with original files"""
    cropped_files = sorted(glob.glob(os.path.join(cropped_dir, '*.nii*')))
    
    file_pairs = []
    for cropped_path in cropped_files:
        basename = os.path.basename(cropped_path)
        subject_id = basename.replace('.nii.gz', '').replace('.nii', '')
        subject_id = subject_id.replace('_cropped', '').replace('_roi', '')
        
        possible_extensions = ['.nii', '.nii.gz']
        original_path = None
        
        for ext in possible_extensions:
            candidate = os.path.join(original_dir, f"{subject_id}{ext}")
            if os.path.exists(candidate):
                original_path = candidate
                break
        
        if original_path is None:
            logger.warning(f"Original not found: {basename}")
            file_pairs.append((cropped_path, None))
        else:
            file_pairs.append((cropped_path, original_path))
    
    return file_pairs


# ========== Main Experiment Class ==========

class JPEG2000Experiment:
    """JPEG2000 compression experiment manager"""
    
    def __init__(self):
        self.processor = BrainImageProcessor()
        self.compressor = JPEG2000Compressor()
        self.evaluator = CompressionEvaluator()
    
    def run_experiment(self, cropped_nifti_path, output_dir, 
                      original_nifti_path=None,
                      cropping_time_info=None,
                      bbox_info=None,
                      quality_levels=[1, 3, 5], axis=2,
                      save_metadata=True):
        """
        Run compression experiment.
        
        Args:
            cropped_nifti_path: path to NIfTI file (ROI or full)
            output_dir: output directory
            original_nifti_path: path to original full file
            cropping_time_info: cropping time
            bbox_info: bbox dict (optional)
            quality_levels: quality levels to test
            axis: slice axis
            save_metadata: save ROI metadata
        """
        cropped_path = Path(cropped_nifti_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cropped data
        cropped_volume, cropped_header, cropped_affine = self.processor.load_nifti(cropped_path)
        if cropped_volume is None:
            return None
        
        cropped_file_size = os.path.getsize(cropped_path)
        cropped_slices = self.processor.extract_slices(cropped_volume, axis=axis, max_slices=None)
        
        # Load original full data
        original_slices = None
        original_affine = None
        original_file_size = None
        original_volume = None
        
        if original_nifti_path and os.path.exists(original_nifti_path):
            original_file_size = os.path.getsize(original_nifti_path)
            
            original_img = nib.load(original_nifti_path)
            original_volume = original_img.get_fdata()
            original_affine = original_img.affine
            
            if original_volume is not None:
                original_slices = self.processor.extract_slices(
                    original_volume, axis=axis, max_slices=None
                )
        else:
            original_affine = cropped_affine
        
        # Create ROI metadata
        roi_metadata = None
        if save_metadata:
            if bbox_info and original_affine is not None:
                roi_metadata = ROIMetadata(
                    bbox=bbox_info,
                    original_shape=original_volume.shape if original_volume is not None else cropped_volume.shape,
                    affine_matrix=original_affine
                )
            elif original_affine is not None:
                roi_metadata = ROIMetadata(
                    bbox=None,
                    original_shape=original_volume.shape if original_volume is not None else cropped_volume.shape,
                    affine_matrix=original_affine
                )
        
        # Calculate 2D bbox
        bbox_2d = None
        if bbox_info:
            if axis == 0:
                bbox_2d = (bbox_info['y_min'], bbox_info['y_max'], 
                          bbox_info['z_min'], bbox_info['z_max'])
            elif axis == 1:
                bbox_2d = (bbox_info['x_min'], bbox_info['x_max'], 
                          bbox_info['z_min'], bbox_info['z_max'])
            else:
                bbox_2d = (bbox_info['x_min'], bbox_info['x_max'], 
                          bbox_info['y_min'], bbox_info['y_max'])
        
        # Cropping time
        if isinstance(cropping_time_info, dict):
            avg_cropping_time = cropping_time_info.get('cropping_time', 0)
        elif isinstance(cropping_time_info, (int, float)):
            avg_cropping_time = cropping_time_info
        else:
            avg_cropping_time = 0
        
        results = {
            'filename': cropped_path.name,
            'original_nifti_path': str(original_nifti_path) if original_nifti_path else 'N/A',
            'cropped_shape': cropped_volume.shape,
            'slice_count': len(cropped_slices),
            'avg_cropping_time': avg_cropping_time,
            'bbox_info': bbox_info,
            'original_file_size': original_file_size,
            'cropped_file_size': cropped_file_size,
            'has_roi_metadata': roi_metadata is not None,
            'metadata_size_bytes': ROIMetadata.TOTAL_SIZE if roi_metadata else 0,
            'experiments': {}
        }
        
        # Test each quality level
        for quality_level in quality_levels:
            compressed_slices_data = []
            slice_metrics = []
            total_comp_time = 0
            total_decomp_time = 0
            
            reconstructed_vis_slice = None
            
            # Compress and evaluate each slice
            for i, cropped_slice in enumerate(cropped_slices):
                try:
                    # Compress
                    compressed, comp_time = self.compressor.compress_image(
                        cropped_slice, quality_level
                    )
                    compressed_slices_data.append(compressed['compressed_data'])
                    total_comp_time += comp_time
                    
                    # Decompress
                    reconstructed, decomp_time = self.compressor.decompress_image([compressed])
                    total_decomp_time += decomp_time
                    
                    
                    # Quality evaluation
                    if original_slices:
                        if bbox_2d:  # ROI mode
                            original_full_slice = original_slices[i]
                            x_min, x_max, y_min, y_max = bbox_2d
                            original_roi = original_full_slice[y_min:y_max, x_min:x_max]
                            reconstructed_roi = reconstructed
                            roi_h, roi_w = original_roi.shape
                            
                            if reconstructed_roi.shape != (roi_h, roi_w):
                                if reconstructed_roi.shape[0] < roi_h or reconstructed_roi.shape[1] < roi_w:
                                    padded = np.zeros((roi_h, roi_w), dtype=reconstructed_roi.dtype)
                                    h_copy = min(reconstructed_roi.shape[0], roi_h)
                                    w_copy = min(reconstructed_roi.shape[1], roi_w)
                                    padded[:h_copy, :w_copy] = reconstructed_roi[:h_copy, :w_copy]
                                    reconstructed_roi = padded
                                else:
                                    reconstructed_roi = reconstructed_roi[:roi_h, :roi_w]
                            
                            mse, psnr, ssim_val = CompressionEvaluator.calculate_metrics(
                                original_roi, reconstructed_roi
                            )
                        else:  # Original-only mode
                            original_slice = original_slices[i]
                            
                            if reconstructed.shape != original_slice.shape:
                                h, w = original_slice.shape
                                if reconstructed.shape[0] < h or reconstructed.shape[1] < w:
                                    padded = np.zeros((h, w), dtype=reconstructed.dtype)
                                    h_copy = min(reconstructed.shape[0], h)
                                    w_copy = min(reconstructed.shape[1], w)
                                    padded[:h_copy, :w_copy] = reconstructed[:h_copy, :w_copy]
                                    reconstructed = padded
                                else:
                                    reconstructed = reconstructed[:h, :w]
                            
                            mse, psnr, ssim_val = CompressionEvaluator.calculate_metrics(
                                original_slice, reconstructed
                            )
                        
                        slice_metrics.append({
                            'slice_index': i,
                            'mse': mse,
                            'psnr': psnr,
                            'ssim': ssim_val,
                            'success': True
                        })
                    
                except Exception as e:
                    logger.warning(f"Slice {i+1} failed: {e}")
                    slice_metrics.append({
                        'slice_index': i,
                        'success': False,
                        'error': str(e)
                    })
            

            # Save compressed volume
            volume_metrics = {}
            
            if compressed_slices_data:
                compressed_file_path = output_dir / f"compressed_q{quality_level}.bin"
                total_compressed_size = self.save_compressed_volume(
                    compressed_slices=compressed_slices_data,
                    roi_metadata=roi_metadata,
                    output_path=compressed_file_path
                )
                
                logger.info(f"Saved: {compressed_file_path} ({total_compressed_size:,} bytes)")
                
                metadata_size = ROIMetadata.TOTAL_SIZE if roi_metadata else 0
                compressed_size_coeffs = total_compressed_size - metadata_size
                
                # Volume-level compression metrics
                if original_file_size:
                    volume_compression_ratio_file = original_file_size / total_compressed_size
                    
                    if original_slices:
                        original_memory_size = sum(s.size * s.itemsize for s in original_slices)
                        volume_compression_ratio_memory = original_memory_size / total_compressed_size
                    else:
                        volume_compression_ratio_memory = None
                    
                    total_pixels = (cropped_volume.shape[0] * 
                                  cropped_volume.shape[1] * 
                                  cropped_volume.shape[2])
                    bpp = (total_compressed_size * 8) / total_pixels
                    
                    volume_metrics = {
                        'compressed_file_size': total_compressed_size,
                        'metadata_size': metadata_size,
                        'compressed_size_coeffs': compressed_size_coeffs,
                        'compression_ratio_file': volume_compression_ratio_file,
                        'compression_ratio_memory': volume_compression_ratio_memory,
                        'bpp': bpp,
                        'total_compression_time': total_comp_time,
                        'total_decompression_time': total_decomp_time,
                        'avg_compression_time_per_slice': total_comp_time / len(cropped_slices),
                        'avg_decompression_time_per_slice': total_decomp_time / len(cropped_slices)
                    }
            
            # Average quality metrics
            successful = [m for m in slice_metrics if m.get('success', False)]
            if successful:
                quality_metrics = {
                    'success_rate': len(successful) / len(cropped_slices),
                    'avg_mse': np.mean([m['mse'] for m in successful]),
                    'avg_psnr': np.mean([m['psnr'] for m in successful]),
                    'avg_ssim': np.mean([m['ssim'] for m in successful])
                }
            else:
                quality_metrics = {'success_rate': 0}
            
            results['experiments'][f'quality_{quality_level}'] = {
                'slice_metrics': slice_metrics,
                'quality_metrics': quality_metrics,
                'volume_metrics': volume_metrics
            }
        
        # Save results
        result_file = output_dir / f"{cropped_path.stem}_results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def save_compressed_volume(self, compressed_slices, roi_metadata, output_path):
        """Save compressed volume with ROI metadata"""
        import struct
        
        # ROI metadata
        if roi_metadata:
            metadata_bytes = roi_metadata.to_bytes()
        else:
            metadata_bytes = b''
        
        # Number of slices
        num_slices = struct.pack('I', len(compressed_slices))
        
        # Slice sizes
        slice_sizes = b''.join([struct.pack('I', len(s)) for s in compressed_slices])
        
        # Write file
        with open(output_path, 'wb') as f:
            f.write(metadata_bytes)
            f.write(num_slices)
            f.write(slice_sizes)
            for s in compressed_slices:
                f.write(s)
        
        total_size = os.path.getsize(output_path)
        
        return total_size
    
    def create_report(self, results, output_dir):
        """Generate experiment report"""
        output_dir = Path(output_dir)
        
        csv_data = []
        for quality_key, quality_data in results['experiments'].items():
            quality_level = quality_key.split('_')[1]
            
            quality_metrics = quality_data.get('quality_metrics', {})
            volume_metrics = quality_data.get('volume_metrics', {})
            
            if quality_metrics.get('success_rate', 0) > 0:
                csv_data.append({
                    'Quality_Level': quality_level,
                    'Success_Rate': f"{quality_metrics['success_rate']:.2%}",
                    'Avg_PSNR_dB': f"{quality_metrics.get('avg_psnr', 0):.2f}",
                    'Avg_SSIM': f"{quality_metrics.get('avg_ssim', 0):.4f}",
                    'Compression_Ratio': f"{volume_metrics.get('compression_ratio_memory', 0):.2f}",
                    'BPP': f"{volume_metrics.get('bpp', 0):.4f}",
                    'Metadata_Size_B': volume_metrics.get('metadata_size', 0),
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_file = output_dir / f"{results['filename']}_summary.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"\nReport saved to {csv_file}")
        
        return df if csv_data else None


# ========== Batch Processing ==========

def batch_process(cropped_dir, original_dir, output_dir, 
                 quality_levels, bbox_csv=None, max_files=None):
    """Batch process ROI files"""
    file_pairs = find_file_pairs(cropped_dir, original_dir)
    
    if not file_pairs:
        logger.error("No file pairs found")
        return
    
    if max_files:
        file_pairs = file_pairs[:max_files]
    
    # Load bbox info
    bbox_dict = {}
    if bbox_csv:
        bbox_dict = load_bbox_info(bbox_csv)
    experiment = JPEG2000Experiment()
    successful = 0
    
    for i, (cropped_path, original_path) in enumerate(file_pairs, 1):
        basename = os.path.basename(cropped_path)
        
        subject_id = basename.replace('.nii.gz', '').replace('.nii', '')
        subject_id = subject_id.replace('_cropped', '').replace('_roi', '')
        
        bbox_info = bbox_dict.get(subject_id)
        
        file_output_dir = Path(output_dir) / Path(cropped_path).stem
        
        try:
            results = experiment.run_experiment(
                cropped_nifti_path=cropped_path,
                output_dir=file_output_dir,
                original_nifti_path=original_path,
                cropping_time_info=0,
                bbox_info=bbox_info,
                quality_levels=quality_levels
            )
            
            if results:
                experiment.create_report(results, file_output_dir)
                successful += 1
                
        except Exception as e:
            logger.error(f"Error: {e}")


def batch_process_original_only(original_dir, output_dir, quality_levels, 
                                max_files=None, axis=2):
    """Batch process original files without ROI"""
    original_files = sorted(glob.glob(os.path.join(original_dir, '*.nii*')))
    
    if not original_files:
        logger.error(f"No NIfTI files found in {original_dir}")
        return
    
    if max_files:
        original_files = original_files[:max_files]
    
    experiment = JPEG2000Experiment()
    successful = 0
    
    for i, original_path in enumerate(original_files, 1):
        basename = os.path.basename(original_path)
        
        file_output_dir = Path(output_dir) / Path(original_path).stem
        
        try:
            results = experiment.run_experiment(
                cropped_nifti_path=original_path,
                output_dir=file_output_dir,
                original_nifti_path=original_path,
                cropping_time_info=0,
                bbox_info=None,
                quality_levels=quality_levels,
                axis=axis,
                save_metadata=False
            )
            
            if results:
                experiment.create_report(results, file_output_dir)
                successful += 1
                
        except Exception as e:
            logger.error(f"Error: {e}")
    
    logger.info(f"\nCompleted: {successful}/{len(original_files)}")


# ========== Command Line Interface ==========

def main():
    parser = argparse.ArgumentParser(
        description='JPEG2000 Medical Image Compression Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ROI mode
  python jpeg.py \\
      --cropped_dir ./cropped_volumes \\
      --original_dir ./nifti \\
      --bbox_csv ./bbox_info.csv \\
      --output ./results 

  # Full mode
  python jpeg.py \\
      --use_original_only \\
      --original_dir ./nifti \\
      --output ./results 
        """
    )
    
    # Data paths
    parser.add_argument('--cropped', help='Single cropped NIfTI file')
    parser.add_argument('--original', help='Single original NIfTI file')
    parser.add_argument('--cropped_dir', help='Directory with cropped files')
    parser.add_argument('--original_dir', help='Directory with original files')
    
    # Output
    parser.add_argument('--output', required=True, help='Output directory')
    
    # Optional
    parser.add_argument('--bbox_csv', help='CSV with 3D bbox info')
    parser.add_argument('--quality_levels', nargs='+', type=int, default=[1, 3, 5],
                       choices=[1,2,3,4,5], help='Quality levels')
    parser.add_argument('--axis', type=int, default=2, choices=[0,1,2],
                       help='Slice axis')
    parser.add_argument('--max_files', type=int, help='Maximum files to process')
    parser.add_argument('--use_original_only', action='store_true',
                       help='Process original files without ROI')
    parser.add_argument('--target_slice', type=int, default=None,
                       help='Target anatomical slice index')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.use_original_only and args.original_dir:
        batch_process_original_only(
            original_dir=args.original_dir,
            output_dir=args.output,
            quality_levels=args.quality_levels,
            max_files=args.max_files,
            axis=args.axis
        )
    
    elif args.cropped:
        experiment = JPEG2000Experiment()
        results = experiment.run_experiment(
            cropped_nifti_path=args.cropped,
            output_dir=args.output,
            original_nifti_path=args.original,
            quality_levels=args.quality_levels,
            axis=args.axis
        )
        if results:
            experiment.create_report(results, args.output)
    
    elif args.cropped_dir:
        if not args.original_dir:
            logger.error("--original_dir required for batch processing")
            sys.exit(1)
        
        batch_process(
            cropped_dir=args.cropped_dir,
            original_dir=args.original_dir,
            output_dir=args.output,
            quality_levels=args.quality_levels,
            bbox_csv=args.bbox_csv,
            max_files=args.max_files
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("JPEG2000 Medical Image Compression Evaluation")
    logger.info("="*70)
    main()