import os
import sys
import numpy as np
import nibabel as nib
import tempfile
from PIL import Image
import pillow_heif
from skimage.metrics import structural_similarity as ssim
from skimage import transform
import json
import argparse
from pathlib import Path
import logging
import time
import pandas as pd
import glob
import ast
import re

# Register HEIF support
pillow_heif.register_heif_opener()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== Image Statistics Analysis ==========

class ImageStatistics:
    """Image statistics analyzer for compression analysis"""
    
    @staticmethod
    def calculate_statistics(image_array):
        """Calculate comprehensive image statistics"""
        stats = {
            'shape': image_array.shape,
            'dtype': str(image_array.dtype),
            'min': float(np.min(image_array)),
            'max': float(np.max(image_array)),
            'mean': float(np.mean(image_array)),
            'std': float(np.std(image_array)),
            'median': float(np.median(image_array)),
            
            # Background ratio (zero or very small values)
            'zero_ratio': float(np.sum(image_array == 0) / image_array.size),
            'low_intensity_ratio': float(np.sum(image_array < 10) / image_array.size),
            
            # Entropy (complexity)
            'entropy': ImageStatistics.calculate_entropy(image_array),
            
            # Histogram statistics
            'histogram': ImageStatistics.get_histogram_stats(image_array),
            
            # Unique values (diversity)
            'unique_values': len(np.unique(image_array)),
            'unique_ratio': len(np.unique(image_array)) / image_array.size,
        }
        return stats
    
    @staticmethod
    def calculate_entropy(image_array):
        """Calculate Shannon entropy"""
        histogram, _ = np.histogram(image_array, bins=256, range=(0, 255))
        histogram = histogram / histogram.sum()
        histogram = histogram[histogram > 0]
        entropy = -np.sum(histogram * np.log2(histogram))
        return float(entropy)
    
    @staticmethod
    def get_histogram_stats(image_array):
        """Calculate histogram statistics"""
        histogram, bins = np.histogram(image_array, bins=256, range=(0, 255))
        return {
            'peak_bin': int(np.argmax(histogram)),
            'peak_count': int(np.max(histogram)),
            'non_zero_bins': int(np.sum(histogram > 0)),
        }


# ========== ROI Metadata Management ==========

class ROIMetadata:
    """
    Metadata class for ROI-based compression.
    
    Stores bounding box coordinates, original volume shape, and affine matrix
    to enable reconstruction of ROI-compressed volumes.
    """
    
    BBOX_SIZE = 12      # 6 values × int16
    SHAPE_SIZE = 6      # 3 values × int16
    AFFINE_SIZE = 36    # 9 values (3×3) × float32
    TOTAL_SIZE = BBOX_SIZE + SHAPE_SIZE + AFFINE_SIZE  # 54 bytes
    
    def __init__(self, bbox=None, original_shape=None, affine_matrix=None):
        """
        Args:
            bbox: dict with keys {x_min, x_max, y_min, y_max, z_min, z_max}
            original_shape: tuple (width, height, depth)
            affine_matrix: 4×4 or 3×3 numpy array
        """
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
    
    def save(self, filepath):
        """Save metadata to file"""
        with open(filepath, 'wb') as f:
            f.write(self.to_bytes())
    
    @classmethod
    def load(cls, filepath):
        """Load metadata from file"""
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.from_bytes(data)


# ========== Utility Functions ==========

def load_bbox_info(bbox_csv_path):
    """Load 3D bounding box information from CSV"""
    if not os.path.exists(bbox_csv_path):
        logger.warning(f"bbox CSV not found: {bbox_csv_path}")
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
        
        logger.info(f"Loaded bbox info for {len(bbox_dict)} subjects")
        return bbox_dict
        
    except Exception as e:
        logger.error(f"Failed to load bbox CSV: {e}")
        return {}


def find_original_nifti(subject_id, original_dir):
    """Find original NIfTI file by subject ID"""
    candidates = [
        os.path.join(original_dir, f"{subject_id}.nii"),
        os.path.join(original_dir, f"{subject_id}.nii.gz"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def find_nifti_files(data_dir, mode='roi', max_files=None):
    """Find NIfTI files in directory"""
    if mode == 'roi':
        cropped_dir = os.path.join(data_dir, 'cropped_volumes')
        if os.path.exists(cropped_dir):
            search_dir = cropped_dir
        else:
            search_dir = data_dir
        
        nifti_files = glob.glob(os.path.join(search_dir, '*.nii.gz'))
        logger.info(f"Searching ROI files in {search_dir}")
    else:
        nifti_files = glob.glob(os.path.join(data_dir, '*.nii'))
        logger.info(f"Searching original files in {data_dir}")
    
    nifti_files = sorted(nifti_files)
    logger.info(f"Found {len(nifti_files)} NIfTI files")
    
    if max_files and len(nifti_files) > max_files:
        nifti_files = nifti_files[:max_files]
        logger.info(f"Limited to first {max_files} files")
    
    return nifti_files


# ========== Image Processing ==========

class BrainImageProcessor:
    """Brain image preprocessing and normalization"""
    
    def __init__(self, normalize_method='minmax', robust_percentiles=(0.5, 99.5)):
        self.normalize_method = normalize_method
        self.robust_percentiles = robust_percentiles

    def compute_volume_minmax(self, volume_data, robust=True):
        """Compute min/max from entire volume for consistent normalization"""
        v = volume_data
        if robust:
            lo, hi = np.percentile(v, self.robust_percentiles)
            return float(lo), float(hi)
        return float(np.min(v)), float(np.max(v))
        
    def load_nifti(self, nii_path):
        """Load NIfTI file"""
        try:
            img = nib.load(nii_path)
            data = img.get_fdata()
            logger.info(f"Loaded: {nii_path}, Shape: {data.shape}")
            return data, img.header, img.affine
        except Exception as e:
            logger.error(f"Error loading {nii_path}: {e}")
            return None, None, None
    
    def normalize_image(self, image_data, fixed_minmax=None):
        """
        Normalize image to [0, 255] range.
        
        Args:
            image_data: input image array
            fixed_minmax: tuple (min, max) for fixed normalization
        """
        if self.normalize_method == 'minmax':
            if fixed_minmax is not None:
                min_val, max_val = fixed_minmax
            else:
                min_val = np.min(image_data)
                max_val = np.max(image_data)

            if max_val > min_val:
                normalized = ((image_data - min_val) / (max_val - min_val) * 255.0)
                normalized = np.clip(normalized, 0, 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(image_data, dtype=np.uint8)

        elif self.normalize_method == 'zscore':
            mean_val = np.mean(image_data)
            std_val = np.std(image_data)
            if std_val > 0:
                z_scored = (image_data - mean_val) / std_val
                normalized = np.clip((z_scored + 3) / 6 * 255, 0, 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(image_data, dtype=np.uint8)

        return normalized
    
    def extract_slices(self, volume_data, axis=2, max_slices=None, fixed_minmax=None):
        """Extract 2D slices from 3D volume with optional fixed normalization"""
        slices = []
        slice_count = volume_data.shape[axis]

        if max_slices is None or max_slices >= slice_count:
            indices = list(range(slice_count))
        else:
            step = max(1, slice_count // max_slices)
            indices = list(range(slice_count // 4, 3 * slice_count // 4, step))[:max_slices]

        for i in indices:
            if axis == 0:
                slice_data = volume_data[i, :, :]
            elif axis == 1:
                slice_data = volume_data[:, i, :]
            else:
                slice_data = volume_data[:, :, i]

            normalized_slice = self.normalize_image(slice_data, fixed_minmax=fixed_minmax)
            slices.append(normalized_slice)

        logger.info(f"Extracted {len(slices)} slices from axis {axis}")
        return slices, indices


# ========== HEIF Compression ==========

class HEIFCompressor:
    """HEIF compression wrapper"""
    
    def __init__(self):
        self.name = "HEIF"
        self._test_setup()
        
    def _test_setup(self):
        """Test HEIF setup"""
        try:
            test_image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            pil_image = Image.fromarray(test_image)
            
            with tempfile.NamedTemporaryFile(suffix='.heic', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            try:
                pil_image.save(temp_path, format='HEIF', quality=80)
                loaded_image = Image.open(temp_path)
                reconstructed = np.array(loaded_image)
                
                if not isinstance(reconstructed, np.ndarray):
                    raise RuntimeError("HEIF reconstruction test failed")
                    
                logger.info("HEIF setup test passed")
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"HEIF setup test failed: {e}")
            raise RuntimeError(f"HEIF not working: {e}")
    
    def compress_image(self, image_array, quality_level=3):
        """
        Compress image using HEIF.
        
        Args:
            image_array: input image (uint8)
            quality_level: 1, 3 (1=lowest, 3=highest)
            
        Returns:
            compressed_data: dict with compressed data
            compression_time: time in seconds
        """
        start_time = time.time()
        
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        
        quality_map = {1: 20, 3: 60}
        heif_quality = quality_map.get(quality_level, 60)
        
        try:
            pil_image = Image.fromarray(image_array)
            
            with tempfile.NamedTemporaryFile(suffix='.heic', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            try:
                pil_image.save(temp_path, format='HEIF', quality=heif_quality)
                
                if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                    raise RuntimeError("HEIF output file is invalid")
                
                compressed_size = os.path.getsize(temp_path)
                with open(temp_path, 'rb') as f:
                    compressed_data = f.read()
                
                compression_time = time.time() - start_time
                
                return {
                    'compressed_data': compressed_data,
                    'original_shape': image_array.shape,
                    'heif_quality': heif_quality,
                    'compressed_size': compressed_size,
                    'method': 'heif'
                }, compression_time
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            raise RuntimeError(f"HEIF compression failed: {e}")
    
    def decompress_image(self, compressed_data):
        """Decompress HEIF image"""
        start_time = time.time()
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.heic', delete=False) as tmp_file:
                temp_path = tmp_file.name
                tmp_file.write(compressed_data[0]['compressed_data'])
                tmp_file.flush()
            
            try:
                pil_image = Image.open(temp_path)
                
                if pil_image.mode != 'L':
                    pil_image = pil_image.convert('L')
                
                reconstructed = np.array(pil_image)
                
                original_shape = compressed_data[0]['original_shape']
                if reconstructed.shape != original_shape:
                    reconstructed = transform.resize(reconstructed, original_shape, 
                                                    preserve_range=True)
                    reconstructed = reconstructed.astype(np.uint8)
                
                decompression_time = time.time() - start_time
                return reconstructed, decompression_time
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            raise RuntimeError(f"HEIF decompression failed: {e}")


# ========== Quality Evaluation ==========

class CompressionEvaluator:
    """Compression quality evaluator with ROI metadata support"""
    
    @staticmethod
    def calculate_mse(original, reconstructed):
        """Calculate Mean Squared Error"""
        if reconstructed is None:
            return float('inf')
        
        if not isinstance(reconstructed, np.ndarray):
            try:
                reconstructed = np.array(reconstructed)
            except Exception:
                return float('inf')
        
        if original.shape != reconstructed.shape:
            try:
                if reconstructed.ndim > 2:
                    if reconstructed.ndim == 3:
                        reconstructed = reconstructed[:, :, 0]
                    elif reconstructed.ndim == 4:
                        reconstructed = reconstructed[0, :, :, 0]
                
                if original.shape != reconstructed.shape:
                    reconstructed = transform.resize(reconstructed, original.shape, 
                                                    preserve_range=True)
                    reconstructed = reconstructed.astype(np.uint8)
                    
            except Exception:
                return float('inf')
        
        try:
            mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
            return mse
        except Exception:
            return float('inf')
    
    @staticmethod
    def calculate_psnr(original, reconstructed, max_pixel_value=255):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = CompressionEvaluator.calculate_mse(original, reconstructed)
        if mse == 0:
            return float('inf')
        if mse == float('inf'):
            return 0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return psnr
    
    @staticmethod
    def calculate_ssim(original, reconstructed):
        """Calculate Structural Similarity Index"""
        if reconstructed is None:
            return 0
        try:
            return ssim(original, reconstructed, data_range=255)
        except:
            return 0
    
    @staticmethod
    def calculate_compression_ratio(original_size, compressed_size):
        """Calculate compression ratio"""
        return original_size / compressed_size if compressed_size > 0 else 0
    
    @staticmethod
    def calculate_bpp(compressed_size, total_pixels):
        """Calculate bits per pixel"""
        return (compressed_size * 8) / total_pixels if total_pixels > 0 else 0
    
    @staticmethod
    def evaluate_slice(original, reconstructed, compressed_size, compression_time, 
                      decompression_time, cropping_time=0, full_original_shape=None, 
                      mode='full', roi_metadata=None):
        """
        Evaluate single slice compression.
        
        Args:
            original: compressed image (ROI in roi mode, full in full mode)
            reconstructed: decompressed image
            compressed_size: HEIF file size in bytes
            compression_time: compression time
            decompression_time: decompression time
            cropping_time: cropping time (roi mode only)
            full_original_shape: original full image shape (roi mode)
            mode: 'full' or 'roi'
            roi_metadata: ROIMetadata object
        """
        
        # Calculate metadata size
        metadata_size = ROIMetadata.TOTAL_SIZE if (mode == 'roi' and roi_metadata is not None) else 0
        
        # Total compressed size = HEIF + metadata
        total_compressed_size = compressed_size + metadata_size
        
        # Actual compressed image info
        actual_compressed_pixels = original.shape[0] * original.shape[1]
        actual_compressed_bytes = original.size * original.itemsize
        
        if mode == 'roi' and full_original_shape is not None:
            # ROI mode: compare with original full size
            full_pixels = full_original_shape[0] * full_original_shape[1]
            full_original_size = full_pixels * 1  # uint8
            
            compression_ratio = CompressionEvaluator.calculate_compression_ratio(
                full_original_size, total_compressed_size)
            bpp = CompressionEvaluator.calculate_bpp(total_compressed_size, full_pixels)
            total_time = cropping_time + compression_time + decompression_time
            
        else:
            # Full mode: use actual image size
            compression_ratio = CompressionEvaluator.calculate_compression_ratio(
                actual_compressed_bytes, total_compressed_size)
            bpp = CompressionEvaluator.calculate_bpp(total_compressed_size, actual_compressed_pixels)
            total_time = compression_time + decompression_time
            full_original_size = actual_compressed_bytes
        
        metrics = {
            'mse': CompressionEvaluator.calculate_mse(original, reconstructed),
            'psnr': CompressionEvaluator.calculate_psnr(original, reconstructed),
            'ssim': CompressionEvaluator.calculate_ssim(original, reconstructed),
            'compression_ratio': compression_ratio,
            'bpp': bpp,
            'original_size': full_original_size,
            'compressed_size': compressed_size,
            'metadata_size': metadata_size,
            'total_compressed_size': total_compressed_size,
            'cropping_time': cropping_time if mode == 'roi' else 0,
            'compression_time': compression_time,
            'decompression_time': decompression_time,
            'total_time': total_time,
            'mode': mode
        }
        
        return metrics


# ========== Main Analysis Class ==========

class HEIFCompressionAnalyzer:
    """HEIF compression analyzer with ROI metadata support"""
    
    def __init__(self, bbox_csv_path=None):
        self.processor = BrainImageProcessor()
        
        try:
            self.heif_compressor = HEIFCompressor()
            self.heif_available = True
            logger.info("HEIF compressor initialized")
        except Exception as e:
            logger.error(f"HEIF initialization failed: {e}")
            self.heif_available = False
            raise RuntimeError("HEIF compression not available")
        
        self.evaluator = CompressionEvaluator()
        
        # Load bbox info
        self.bbox_dict = {}
        if bbox_csv_path and os.path.exists(bbox_csv_path):
            self.bbox_dict = load_bbox_info(bbox_csv_path)
        
    def analyze_heif_compression(self, input_path, output_dir, quality_levels=[1,3,5],
                                axis=2, mode='roi', original_dir=None):
        """
        Analyze HEIF compression on NIfTI volume.
        
        Args:
            input_path: path to NIfTI file
            output_dir: output directory
            quality_levels: HEIF quality levels to test
            axis: slice axis
            mode: 'full' or 'roi'
            original_dir: directory with original files (for ROI mode)
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load input volume
        volume_data, header, affine_matrix = self.processor.load_nifti(input_path)
        if volume_data is None:
            return None

        # Extract subject ID
        subject_id = input_path.stem.replace('.nii', '').replace('_cropped', '').replace('_roi', '')

        # ROI mode: load original for normalization
        fixed_minmax = None
        original_full_path = None
        full_vol_shape = None

        if mode == 'roi':
            if original_dir:
                original_full_path = find_original_nifti(subject_id, original_dir)
            
            if original_full_path and os.path.exists(original_full_path):
                full_vol, _, _ = self.processor.load_nifti(original_full_path)
                full_vol_shape = full_vol.shape 
                fixed_minmax = self.processor.compute_volume_minmax(full_vol, robust=True)
                logger.info(f"Using normalization from original: {Path(original_full_path).name}")

        # Extract slices
        slices, slice_indices = self.processor.extract_slices(
            volume_data, axis=axis, max_slices=None, fixed_minmax=fixed_minmax
        )

        # Load ROI information
        bbox_info = None
        roi_metadata = None

        if mode == 'roi':
            bbox_info = self.bbox_dict.get(subject_id)

            if bbox_info and affine_matrix is not None and full_vol_shape is not None:
                roi_metadata = ROIMetadata(
                    bbox=bbox_info,
                    original_shape=full_vol_shape,
                    affine_matrix=affine_matrix
                )
                logger.info(f"ROI metadata created ({ROIMetadata.TOTAL_SIZE} bytes)")

        results = {
            'filename': input_path.name,
            'subject_id': subject_id,
            'mode': mode,
            'original_shape': volume_data.shape,
            'slice_count': len(slices),
            'quality_levels': quality_levels,
            'axis': axis,
            'has_roi_metadata': roi_metadata is not None,
            'metadata_size_bytes': ROIMetadata.TOTAL_SIZE if roi_metadata else 0,
            'heif_results': {}
        }
        
        logger.info(f"Analyzing {len(slices)} slices in {mode.upper()} mode...")
        
        # Collect statistics
        slice_stats = []
        
        for quality_level in quality_levels:
            recon_dir = output_dir / f"recon_{mode}" / f"Q{quality_level}"
            recon_dir.mkdir(parents=True, exist_ok=True)
            
            original_dir_path = output_dir / f"original_{mode}" / f"Q{quality_level}"
            original_dir_path.mkdir(parents=True, exist_ok=True)
            
            slice_metrics = []
            slice_failures = 0
            
            for i, (slice_data, orig_idx) in enumerate(zip(slices, slice_indices)):
                try:
                    # Collect statistics
                    stats_before = ImageStatistics.calculate_statistics(slice_data)
                    stats_before['mode'] = mode
                    stats_before['slice_index'] = orig_idx
                    stats_before['quality_level'] = quality_level
                    slice_stats.append(stats_before)
                    
                    # Compress
                    compressed, comp_time = self.heif_compressor.compress_image(
                        slice_data, quality_level
                    )
                    
                    # Decompress
                    reconstructed, decomp_time = self.heif_compressor.decompress_image(
                        [compressed]
                    )
                    
                    # Evaluate
                    compressed_size = compressed['compressed_size']
                    
                    if mode == 'roi' and bbox_info and full_vol_shape:
                        # Calculate original slice shape
                        if axis == 0:
                            full_slice_shape = (full_vol_shape[1], full_vol_shape[2])
                        elif axis == 1:
                            full_slice_shape = (full_vol_shape[0], full_vol_shape[2])
                        else:
                            full_slice_shape = (full_vol_shape[0], full_vol_shape[1])
                        
                        metrics = self.evaluator.evaluate_slice(
                            slice_data, reconstructed, compressed_size, 
                            comp_time, decomp_time,
                            cropping_time=0,
                            full_original_shape=full_slice_shape,
                            mode='roi',
                            roi_metadata=roi_metadata
                        )
                    else:
                        metrics = self.evaluator.evaluate_slice(
                            slice_data, reconstructed, compressed_size, 
                            comp_time, decomp_time,
                            mode='full',
                            roi_metadata=None
                        )
                    
                    metrics['slice_index'] = i
                    metrics['success'] = True
                    metrics['heif_quality'] = compressed['heif_quality']
                    slice_metrics.append(metrics)
                    
                    
                except Exception as e:
                    logger.warning(f"Slice {i} failed: {e}")
                    slice_failures += 1
                    slice_metrics.append({
                        'slice_index': i,
                        'success': False,
                        'error': str(e),
                        'mode': mode
                    })
            
            # Calculate averages
            successful_metrics = [m for m in slice_metrics if m.get('success', False)]
            
            if successful_metrics:
                avg_metrics = {
                    'successful_slices': len(successful_metrics),
                    'failed_slices': slice_failures,
                    'success_rate': len(successful_metrics) / len(slices),
                    'avg_mse': np.mean([m['mse'] for m in successful_metrics]),
                    'avg_psnr': np.mean([m['psnr'] for m in successful_metrics if np.isfinite(m['psnr'])]),
                    'avg_ssim': np.mean([m['ssim'] for m in successful_metrics]),
                    'avg_compression_ratio': np.mean([m['compression_ratio'] for m in successful_metrics]),
                    'avg_bpp': np.mean([m['bpp'] for m in successful_metrics]),
                    'avg_metadata_size': np.mean([m['metadata_size'] for m in successful_metrics]),
                    'avg_compression_time': np.mean([m['compression_time'] for m in successful_metrics]),
                    'avg_decompression_time': np.mean([m['decompression_time'] for m in successful_metrics]),
                    'avg_total_time': np.mean([m['total_time'] for m in successful_metrics]),
                }
            else:
                avg_metrics = {
                    'successful_slices': 0,
                    'failed_slices': len(slices),
                    'success_rate': 0,
                }
            
            results['heif_results'][f'quality_{quality_level}'] = {
                'slice_metrics': slice_metrics,
                'average_metrics': avg_metrics
            }
            
            logger.info(f"Q{quality_level}: {len(successful_metrics)}/{len(slices)} successful")
        
        # Save statistics
        if slice_stats:
            stats_file = output_dir / f"{input_path.stem}_stats_{mode}.json"
            with open(stats_file, 'w') as f:
                json.dump(slice_stats, f, indent=2)
        
        # Save results
        result_file = output_dir / f"{input_path.stem}_heif_{mode}_analysis.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {result_file}")
        return results
    
    def create_analysis_report(self, results, output_dir):
        """Generate analysis report"""
        output_dir = Path(output_dir)
        mode = results.get('mode', 'unknown')
        
        csv_data = []
        for quality_key, quality_data in results['heif_results'].items():
            quality_level = quality_key.split('_')[1]
            avg_metrics = quality_data['average_metrics']
            
            row = {
                'Mode': mode.upper(),
                'Quality_Level': quality_level,
                'Success_Rate': f"{avg_metrics.get('success_rate', 0):.2%}",
            }
            
            if avg_metrics.get('success_rate', 0) > 0:
                row.update({
                    'Avg_PSNR': f"{avg_metrics['avg_psnr']:.2f}",
                    'Avg_SSIM': f"{avg_metrics['avg_ssim']:.4f}",
                    'Compression_Ratio': f"{avg_metrics['avg_compression_ratio']:.2f}",
                    'BPP': f"{avg_metrics['avg_bpp']:.4f}",
                    'Metadata_Size_B': f"{avg_metrics.get('avg_metadata_size', 0):.0f}",
                })
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        
        filename = results.get('filename', 'unknown')
        summary_file = output_dir / f"{filename}_summary.csv"
        df.to_csv(summary_file, index=False)
        
        logger.info(f"Report saved: {summary_file}")
        logger.info(f"\n{df.to_string(index=False)}")
        
        return df


# ========== Batch Processing ==========

def batch_process(input_dir, output_dir, quality_levels, mode='roi', 
                 max_files=None, original_dir=None, bbox_csv=None):
    """Batch process NIfTI files"""
    all_files = find_nifti_files(input_dir, mode=mode, max_files=max_files)
    
    if not all_files:
        logger.error(f"No NIfTI files found in {input_dir}")
        return
    
    analyzer = HEIFCompressionAnalyzer(bbox_csv_path=bbox_csv)
    results_list = []
    
    successful = 0
    failed = 0
    
    for i, file_path in enumerate(all_files, 1):
        logger.info(f"\n[{i}/{len(all_files)}] Processing: {os.path.basename(file_path)}")
        
        if mode == 'full':
            base_name = Path(file_path).stem
            file_output_dir = Path(output_dir) / base_name
        else:
            filename = Path(file_path).name
            dir_name = filename.replace('.gz', '')
            file_output_dir = Path(output_dir) / dir_name
        
        try:
            results = analyzer.analyze_heif_compression(
                input_path=file_path,
                output_dir=file_output_dir,
                quality_levels=quality_levels,
                mode=mode, 
                original_dir=original_dir 
            )
            
            if results:
                analyzer.create_analysis_report(results, file_output_dir)
                results_list.append({
                    'filename': os.path.basename(file_path),
                    'mode': mode,
                    'status': 'success',
                    'has_metadata': results.get('has_roi_metadata', False),
                })
                successful += 1
            else:
                results_list.append({
                    'filename': os.path.basename(file_path),
                    'mode': mode,
                    'status': 'failed',
                    'has_metadata': False,
                })
                failed += 1
                
        except Exception as e:
            results_list.append({
                'filename': os.path.basename(file_path),
                'mode': mode,
                'status': 'error',
                'has_metadata': False,
                'error': str(e)
            })
            failed += 1
            logger.error(f"Error: {e}")
    
    # Summary
    with_metadata = sum(1 for r in results_list if r.get('has_metadata', False))
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Batch processing completed ({mode.upper()} mode)")
    logger.info(f"Success: {successful}/{len(all_files)}")
    logger.info(f"ROI Metadata: {with_metadata}/{successful}")
    logger.info(f"{'='*60}")


# ========== Command Line Interface ==========

def main():
    parser = argparse.ArgumentParser(
        description='HEIF Medical Image Compression Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ROI mode (cropped images with metadata)
  python HEIF.py \\
      --mode roi \\
      --input_dir ./cropped_volumes \\
      --output ./results/roi \\
      --bbox_csv ./bbox_info.csv \\
      --original_dir ./original_nifti

  # Full mode (original images without metadata)
  python HEIF.py \\
      --mode full \\
      --input_dir ./original_nifti \\
      --output ./results/full

  # Single file
  python HEIF.py \\
      --mode roi \\
      --input ./file.nii.gz \\
      --output ./results \\
      --original_dir ./original_nifti
        """
    )
    
    # Required arguments
    parser.add_argument('--mode', choices=['full', 'roi'], required=True,
                       help='Compression mode')
    parser.add_argument('--output', required=True,
                       help='Output directory')
    
    # Data paths
    parser.add_argument('--input',
                       help='Single NIfTI file path')
    parser.add_argument('--input_dir',
                       help='Directory with NIfTI files')
    parser.add_argument('--original_dir',
                       help='Directory with original files (for ROI normalization)')
    parser.add_argument('--bbox_csv',
                       help='CSV file with 3D bbox info')
    
    # Processing options
    parser.add_argument('--quality_levels', nargs='+', type=int, default=[1, 3],
                       choices=[1,2,3,4,5],
                       help='Quality levels to test')
    parser.add_argument('--axis', type=int, default=2, choices=[0,1,2],
                       help='Slice axis: 0=sagittal, 1=coronal, 2=axial')
    parser.add_argument('--max_files', type=int,
                       help='Maximum number of files to process')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.input:
        # Single file
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            return
        
        analyzer = HEIFCompressionAnalyzer(bbox_csv_path=args.bbox_csv)
        
        try:
            results = analyzer.analyze_heif_compression(
                input_path=args.input,
                output_dir=args.output,
                quality_levels=args.quality_levels,
                axis=args.axis,
                mode=args.mode,
                original_dir=args.original_dir
            )
            
            if results:
                analyzer.create_analysis_report(results, args.output)
                logger.info("Processing completed")
            else:
                logger.error("Processing failed")
                
        except Exception as e:
            logger.error(f"Error: {e}")
    
    elif args.input_dir:
        # Batch processing
        if not os.path.exists(args.input_dir):
            logger.error(f"Input directory not found: {args.input_dir}")
            return
        
        try:
            batch_process(
                input_dir=args.input_dir,
                output_dir=args.output,
                quality_levels=args.quality_levels,
                mode=args.mode,
                max_files=args.max_files,
                original_dir=args.original_dir,
                bbox_csv=args.bbox_csv
            )
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("HEIF Medical Image Compression Evaluation")
    logger.info("="*70)
    
    if len(sys.argv) == 1:
        print("\nExamples:")
        print("\n1. ROI mode:")
        print("   python eval_heif.py --mode roi --input_dir ./cropped --output ./results")
        print("\n2. Full mode:")
        print("   python eval_heif.py --mode full --input_dir ./original --output ./results")
        print("\nFor help: python eval_heif.py --help")
    else:
        main()