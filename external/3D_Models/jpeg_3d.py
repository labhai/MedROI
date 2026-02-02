"""
3D Wavelet Medical Image Compression Evaluation with ROI Support

This script evaluates 3D Wavelet compression on medical imaging datasets
using PyWavelets library with support for ROI-based compression and metadata.

Features:
- 3D Wavelet compression using PyWavelets (bior4.4 default)
- ROI (Region of Interest) compression with metadata
- Multiple quality levels with quantization
- Quality metrics: PSNR, SSIM (3D)
- Compression metrics: BPP, Compression Ratio
- Lossless metadata preservation

Reference: JPEG2000 standard wavelet (bior4.4)
Author: [Your Name]
License: MIT
"""

import os
import sys
import numpy as np
import nibabel as nib
from skimage.metrics import structural_similarity as ssim
import json
import argparse
from pathlib import Path
import logging
import time
import pandas as pd
import glob
import pickle
import gzip
import cv2
from PIL import Image

# PyWavelets
try:
    import pywt
    logging.info("PyWavelets imported successfully")
except ImportError:
    print("PyWavelets not available. Install with: pip install PyWavelets")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    
    def normalize_volume(self, volume_data):
        """Normalize 3D volume to [0, 255] range"""
        if self.normalize_method == 'minmax':
            min_val, max_val = np.min(volume_data), np.max(volume_data)
            if max_val > min_val:
                return ((volume_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        elif self.normalize_method == 'zscore':
            mean_val, std_val = np.mean(volume_data), np.std(volume_data)
            if std_val > 0:
                z_scored = (volume_data - mean_val) / std_val
                return np.clip((z_scored + 3) / 6 * 255, 0, 255).astype(np.uint8)
        
        return np.zeros_like(volume_data, dtype=np.uint8)


# ========== 3D Wavelet Compression ==========

class Wavelet3DCompressor:
    """3D Wavelet compressor with ROI metadata support"""
    
    # Quality level to quantization step mapping
    QUANTIZATION_STEPS = {
        1: 1.0,    # Highest quality (near-lossless)
        2: 2.0,    # High quality
        3: 5.0,    # Medium quality
        4: 10.0,   # Low quality
        5: 20.0    # Lowest quality (high compression)
    }
    
    def __init__(self, wavelet='bior4.4', level=3):
        """
        Args:
            wavelet: wavelet type (JPEG2000 standard: bior4.4)
            level: wavelet decomposition level
        """
        self.wavelet = wavelet
        self.level = level
        
    def compress_volume(self, volume_array, quality_level=3, roi_metadata=None):
        """
        Compress 3D volume with ROI metadata.
        
        Args:
            volume_array: 3D numpy array (uint8)
            quality_level: 1-5 (1=best quality, 5=best compression)
            roi_metadata: ROIMetadata object (for ROI compression)
            
        Returns:
            compressed_data: dict with compressed data
            comp_time: compression time
        """
        start_time = time.time()
        
        if volume_array.dtype != np.uint8:
            volume_array = volume_array.astype(np.uint8)
        
        try:
            # 3D wavelet decomposition
            coeffs = pywt.wavedecn(volume_array.astype(np.float32), 
                                   wavelet=self.wavelet, 
                                   level=self.level,
                                   mode='periodization')
            
            # Quantization
            quant_step = self.QUANTIZATION_STEPS.get(quality_level, 5.0)
            coeffs_quantized = self._quantize_coeffs(coeffs, quant_step)
            
            # Serialization and compression
            serialized = pickle.dumps(coeffs_quantized)
            compressed_coeffs = gzip.compress(serialized, compresslevel=9)
            
            # Add ROI metadata
            metadata_bytes = b''
            metadata_size = 0
            has_roi_metadata = False
            
            if roi_metadata is not None:
                metadata_bytes = roi_metadata.to_bytes()
                metadata_size = len(metadata_bytes)
                has_roi_metadata = True
            
            # Final compressed data = metadata + coeffs
            compressed_data = metadata_bytes + compressed_coeffs
            
            comp_time = time.time() - start_time
            
            compressed_size_total = len(compressed_data)
            compressed_size_coeffs = len(compressed_coeffs)
            original_size = volume_array.size * volume_array.itemsize
            
            return {
                'compressed_data': compressed_data,
                'original_shape': volume_array.shape,
                'compressed_size_total': compressed_size_total,
                'compressed_size_coeffs': compressed_size_coeffs,
                'metadata_size': metadata_size,
                'has_roi_metadata': has_roi_metadata,
                'quality_level': quality_level,
                'quant_step': quant_step,
                'wavelet': self.wavelet,
                'level': self.level,
                'method': f'wavelet3d_q{quality_level}{"_roi" if has_roi_metadata else ""}'
            }, comp_time
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise
    
    def decompress_volume(self, compressed_data, return_full_if_roi=False,
                         original_full_volume=None):
        """
        Decompress 3D volume with ROI metadata.
        
        Args:
            compressed_data: output from compress_volume
            return_full_if_roi: pad ROI to full size
            original_full_volume: original full volume for padding
            
        Returns:
            reconstructed: reconstructed 3D array
            decomp_time: decompression time
            roi_metadata: restored ROI metadata (if exists)
            full_reconstructed: full-size padded volume (if requested)
        """
        start_time = time.time()
        
        try:
            # Separate metadata
            roi_metadata = None
            compressed_bytes = compressed_data['compressed_data']
            
            if compressed_data.get('has_roi_metadata', False):
                metadata_size = compressed_data['metadata_size']
                metadata_bytes = compressed_bytes[:metadata_size]
                coeffs_bytes = compressed_bytes[metadata_size:]
                
                roi_metadata = ROIMetadata.from_bytes(metadata_bytes)
            else:
                coeffs_bytes = compressed_bytes
            
            # Decompress and deserialize
            decompressed = gzip.decompress(coeffs_bytes)
            coeffs_quantized = pickle.loads(decompressed)
            
            # Dequantization
            quant_step = compressed_data['quant_step']
            coeffs = self._dequantize_coeffs(coeffs_quantized, quant_step)
            
            # 3D wavelet reconstruction
            reconstructed = pywt.waverecn(coeffs, 
                                         wavelet=compressed_data['wavelet'],
                                         mode='periodization')
            
            # Crop to original shape
            original_shape = compressed_data['original_shape']
            reconstructed = reconstructed[:original_shape[0], 
                                        :original_shape[1], 
                                        :original_shape[2]]
            
            # ROI padding
            full_reconstructed = None
            if return_full_if_roi and roi_metadata is not None and roi_metadata.bbox is not None:
                try:
                    full_reconstructed = pad_roi_volume_to_full(
                        reconstructed, 
                        roi_metadata,
                        original_full_volume=original_full_volume
                    )
                except Exception as e:
                    logger.warning(f"Padding failed: {e}")
                    full_reconstructed = None

            decomp_time = time.time() - start_time
            
            return reconstructed, decomp_time, roi_metadata, full_reconstructed
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise
    
    def _quantize_coeffs(self, coeffs, quant_step):
        """Quantize wavelet coefficients"""
        coeffs_quantized = []
        
        # Approximation coefficients
        cA = coeffs[0]
        coeffs_quantized.append(np.round(cA / quant_step).astype(np.int16))
        
        # Detail coefficients
        for detail_dict in coeffs[1:]:
            quantized_dict = {}
            for key, coeff in detail_dict.items():
                quantized_dict[key] = np.round(coeff / quant_step).astype(np.int16)
            coeffs_quantized.append(quantized_dict)
        
        return coeffs_quantized
    
    def _dequantize_coeffs(self, coeffs_quantized, quant_step):
        """Dequantize wavelet coefficients"""
        coeffs = []
        
        # Approximation coefficients
        cA = coeffs_quantized[0].astype(np.float32) * quant_step
        coeffs.append(cA)
        
        # Detail coefficients
        for detail_dict in coeffs_quantized[1:]:
            dequantized_dict = {}
            for key, coeff in detail_dict.items():
                dequantized_dict[key] = coeff.astype(np.float32) * quant_step
            coeffs.append(dequantized_dict)
        
        return coeffs


# ========== Quality Evaluation ==========

class CompressionEvaluator:
    """Compression quality evaluator for 3D volumes"""
    
    @staticmethod
    def calculate_metrics_3d(original, reconstructed):
        """Calculate 3D volume quality metrics"""
        # MSE
        mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
        
        # PSNR
        if mse == 0:
            psnr = 100.0
        elif mse < 1e-10:
            psnr = 20 * np.log10(255 / np.sqrt(mse))
        else:
            psnr = 20 * np.log10(255 / np.sqrt(mse))
        
        # SSIM (slice-wise average)
        ssim_values = []
        for i in range(original.shape[2]):
            slice_orig = original[:, :, i]
            slice_recon = reconstructed[:, :, i]
            if slice_orig.std() > 0 and slice_recon.std() > 0:
                ssim_val = ssim(slice_orig, slice_recon, data_range=255)
                ssim_values.append(ssim_val)
        
        avg_ssim = np.mean(ssim_values) if ssim_values else 0.0
        
        return mse, psnr, avg_ssim
    
    @staticmethod
    def evaluate_volume(original_cropped_volume, reconstructed, 
                       compressed_size_total, compressed_size_coeffs, metadata_size,
                       comp_time, decomp_time,
                       original_full_volume=None, cropping_time=0,
                       original_full_file_path=None):
        """
        Evaluate 3D volume compression.
        
        Args:
            original_cropped_volume: cropped ROI volume (for quality)
            reconstructed: decompressed volume
            compressed_size_total: total compressed size (metadata + coeffs)
            compressed_size_coeffs: coefficients compressed size
            metadata_size: ROI metadata size
            comp_time: compression time
            decomp_time: decompression time
            original_full_volume: original full volume (for size calculation)
            cropping_time: ROI cropping time
            original_full_file_path: original file path
        """
        # Quality metrics
        mse, psnr, ssim_val = CompressionEvaluator.calculate_metrics_3d(
            original_cropped_volume, reconstructed
        )
        
        # Memory-based compression ratio
        if original_full_volume is not None:
            original_memory_size = original_full_volume.size * original_full_volume.itemsize
        else:
            logger.warning("Using cropped size for compression ratio")
            original_memory_size = original_cropped_volume.size * original_cropped_volume.itemsize
        
        compression_ratio_memory = original_memory_size / compressed_size_total if compressed_size_total > 0 else 0
        
        # File-based compression ratio
        compression_ratio_file = None
        original_file_size = None
        
        if original_full_file_path and os.path.exists(original_full_file_path):
            original_file_size = os.path.getsize(original_full_file_path)
            compression_ratio_file = original_file_size / compressed_size_total if compressed_size_total > 0 else 0
        
        # BPP
        total_voxels = original_full_volume.size if original_full_volume is not None else original_cropped_volume.size
        bpp = (compressed_size_total * 8) / total_voxels
        
        # Total time
        total_processing_time = cropping_time + comp_time + decomp_time
        
        return {
            # Quality metrics
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim_val,
            
            # Compression metrics
            'compression_ratio': compression_ratio_memory,
            'compression_ratio_file_based': compression_ratio_file,
            'bpp': bpp,
            
            # Size info
            'original_full_size': original_memory_size,
            'original_file_size': original_file_size,
            'cropped_size': original_cropped_volume.size * original_cropped_volume.itemsize,
            'compressed_size_total': compressed_size_total,
            'compressed_size_coeffs': compressed_size_coeffs,
            'metadata_size': metadata_size,
            'metadata_overhead_pct': (metadata_size / compressed_size_total * 100) if compressed_size_total > 0 else 0,
            
            # Time info
            'cropping_time': cropping_time,
            'compression_time': comp_time,
            'decompression_time': decomp_time,
            'total_time': total_processing_time
        }


# ========== Utility Functions ==========

def pad_roi_volume_to_full(recon_roi_vol, roi_metadata, original_full_volume=None):
    """
    Pad ROI volume to original full size.
    
    Args:
        recon_roi_vol: reconstructed ROI volume
        roi_metadata: ROI metadata with bbox
        original_full_volume: actual loaded original volume (priority)
        
    Returns:
        canvas: full-size padded volume
    """
    assert recon_roi_vol.ndim == 3
    assert roi_metadata is not None and roi_metadata.bbox is not None

    bbox = roi_metadata.bbox
    x0, x1 = int(bbox['x_min']), int(bbox['x_max'])
    y0, y1 = int(bbox['y_min']), int(bbox['y_max'])
    z0, z1 = int(bbox['z_min']), int(bbox['z_max'])

    # Use actual loaded volume shape if available
    if original_full_volume is not None:
        H, W, D = original_full_volume.shape
    else:
        assert roi_metadata.original_shape is not None
        H, W, D = map(int, roi_metadata.original_shape)
        logger.warning(f"Using metadata shape (fallback): ({H}, {W}, {D})")

    canvas = np.zeros((H, W, D), dtype=recon_roi_vol.dtype)

    expected_x = x1 - x0 + 1
    expected_y = y1 - y0 + 1
    expected_z = z1 - z0 + 1

    # Shape verification and correction
    if recon_roi_vol.shape != (expected_x, expected_y, expected_z):
        logger.warning(f"Shape mismatch: {recon_roi_vol.shape} vs ({expected_x}, {expected_y}, {expected_z})")
        
        resized = np.zeros((expected_x, expected_y, expected_z), dtype=recon_roi_vol.dtype)
        z_min = min(expected_z, recon_roi_vol.shape[2])
        for zi in range(z_min):
            resized[:, :, zi] = cv2.resize(
                recon_roi_vol[:, :, zi], 
                (expected_y, expected_x),
                interpolation=cv2.INTER_NEAREST
            )
        vol = resized
    else:
        vol = recon_roi_vol

    canvas[x0:x1+1, y0:y1+1, z0:z1+1] = vol
    
    return canvas


def load_bbox_info(bbox_csv_path):
    """Load bounding box information from CSV"""
    if not os.path.exists(bbox_csv_path):
        logger.warning(f"BBox file not found: {bbox_csv_path}")
        return {}
    
    bbox_dict = {}
    
    try:
        df = pd.read_csv(bbox_csv_path)
        
        bbox_columns = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']
        has_all_bbox_cols = all(col in df.columns for col in bbox_columns)
        
        if has_all_bbox_cols:
            logger.info(f"BBox info found in {bbox_csv_path}")
            
            key_column = None
            if 'filename' in df.columns:
                key_column = 'filename'
            elif 'subject_id' in df.columns:
                key_column = 'subject_id'
            elif 'subject' in df.columns:
                key_column = 'subject'
            else:
                key_column = df.columns[0]
            
            for _, row in df.iterrows():
                key = str(row[key_column])
                
                key_variants = [
                    key + '.nii.gz',
                    key + '.nii',
                    key + '_cropped.nii.gz',
                    key + '_cropped.nii'
                ]
                
                bbox_info = {
                    'x_min': int(row['x_min']),
                    'x_max': int(row['x_max']),
                    'y_min': int(row['y_min']),
                    'y_max': int(row['y_max']),
                    'z_min': int(row['z_min']),
                    'z_max': int(row['z_max'])
                }
                
                for variant in key_variants:
                    bbox_dict[variant] = bbox_info
        else:
            logger.warning(f"Missing BBox columns in {bbox_csv_path}")
                
    except Exception as e:
        logger.warning(f"Failed to load BBox: {e}")
    
    return bbox_dict


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

class Wavelet3DExperiment:
    """3D Wavelet compression experiment manager"""
    
    def __init__(self, wavelet='bior4.4', level=3):
        self.processor = BrainImageProcessor()
        self.compressor = Wavelet3DCompressor(wavelet=wavelet, level=level)
        self.evaluator = CompressionEvaluator()
        
    def run_experiment(self, cropped_nifti_path, output_dir,
                      original_nifti_path=None,
                      cropping_time_info=None,
                      quality_levels=[1, 3, 5],
                      bbox_info=None):
        """
        Run compression experiment.
        
        Args:
            cropped_nifti_path: path to NIfTI file (ROI or full)
            output_dir: output directory
            original_nifti_path: path to original full file
            cropping_time_info: cropping time
            quality_levels: quality levels to test
            bbox_info: bbox dict (optional)
        """
        cropped_path = Path(cropped_nifti_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cropped_file_size = os.path.getsize(cropped_path)
        
        # Load cropped data
        cropped_volume, cropped_header, cropped_affine = self.processor.load_nifti(cropped_path)
        if cropped_volume is None:
            return None
        
        cropped_volume_normalized = self.processor.normalize_volume(cropped_volume)
        
        # Load original full data
        original_volume = None
        original_volume_normalized = None
        
        if original_nifti_path and os.path.exists(original_nifti_path):
            original_volume, original_header, original_affine = self.processor.load_nifti(original_nifti_path)
            if original_volume is not None:
                original_volume_normalized = self.processor.normalize_volume(original_volume)
        else:
            original_affine = cropped_affine
        
        # Create ROI metadata
        roi_metadata = None
        if bbox_info and original_affine is not None:
            roi_metadata = ROIMetadata(
                bbox=bbox_info,
                original_shape=original_volume.shape if original_volume is not None else cropped_volume.shape,
                affine_matrix=original_affine
            )
        
        # Cropping time
        if isinstance(cropping_time_info, dict):
            cropping_time = cropping_time_info.get('cropping_time', 0)
        elif isinstance(cropping_time_info, (int, float)):
            cropping_time = cropping_time_info
        else:
            cropping_time = 0
        
        results = {
            'filename': cropped_path.name,
            'original_nifti_path': str(original_nifti_path) if original_nifti_path else 'N/A',
            'cropped_shape': cropped_volume_normalized.shape,
            'compression_method': '3D_Wavelet',
            'wavelet_type': self.compressor.wavelet,
            'wavelet_level': self.compressor.level,
            'has_roi_metadata': roi_metadata is not None,
            'metadata_size_bytes': ROIMetadata.TOTAL_SIZE if roi_metadata else 0,
            'cropping_time': cropping_time,
            'experiments': {}
        }
        
        # Test each quality level
        for quality_level in quality_levels:
            
            try:
                # Compress
                compressed, comp_time = self.compressor.compress_volume(
                    cropped_volume_normalized, quality_level, roi_metadata=roi_metadata
                )
                
                # Decompress
                reconstructed, decomp_time, restored_metadata, full_reconstructed = self.compressor.decompress_volume(
                    compressed, 
                    return_full_if_roi=True,
                    original_full_volume=original_volume_normalized
                )
                
                # Evaluate
                metrics = self.evaluator.evaluate_volume(
                    original_cropped_volume=cropped_volume_normalized,
                    reconstructed=reconstructed,
                    compressed_size_total=compressed['compressed_size_total'],
                    compressed_size_coeffs=compressed['compressed_size_coeffs'],
                    metadata_size=compressed['metadata_size'],
                    comp_time=comp_time,
                    decomp_time=decomp_time,
                    original_full_volume=original_volume_normalized,
                    cropping_time=cropping_time,
                    original_full_file_path=original_nifti_path
                )
                
                metrics['success'] = True
                
            except Exception as e:
                logger.error(f"Quality {quality_level} failed: {e}")
                metrics = {
                    'success': False,
                    'error': str(e)
                }
            
            results['experiments'][f'quality_{quality_level}'] = metrics
        
        # Save results
        result_file = output_dir / f"{cropped_path.stem}_results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        
        return results
    
    def create_report(self, results, output_dir):
        """Generate experiment report"""
        output_dir = Path(output_dir)
        
        csv_data = []
        for quality_key, metrics in results['experiments'].items():
            quality_level = quality_key.split('_')[1]
            
            if metrics.get('success', False):
                csv_data.append({
                    'Quality_Level': quality_level,
                    'PSNR_dB': f"{metrics.get('psnr', 0):.2f}",
                    'SSIM': f"{metrics.get('ssim', 0):.4f}",
                    'Compression_Ratio': f"{metrics.get('compression_ratio', 0):.2f}",
                    'BPP': f"{metrics.get('bpp', 0):.4f}",
                    'Metadata_Size_B': metrics.get('metadata_size', 0),
                    'Total_Time_s': f"{metrics.get('total_time', 0):.4f}",
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_file = output_dir / f"{results['filename']}_summary.csv"
            df.to_csv(csv_file, index=False)
            
        else:
            logger.warning("No successful experiments to report")
        
        return df if csv_data else None


# ========== Batch Processing ==========

def batch_process(cropped_dir, original_dir, output_dir,
                 quality_levels, bbox_file=None, 
                 max_files=None, wavelet='bior4.4', level=3):
    """Batch process ROI files"""
    file_pairs = find_file_pairs(cropped_dir, original_dir)
    
    if not file_pairs:
        logger.error("No file pairs found")
        return
    
    if max_files:
        file_pairs = file_pairs[:max_files]
    
    bbox_info_dict = {}
    
    if bbox_file:
        bbox_info_dict = load_bbox_info(bbox_file)
    
    experiment = Wavelet3DExperiment(wavelet=wavelet, level=level)
    successful = 0
    
    for i, (cropped_path, original_path) in enumerate(file_pairs, 1):

        file_output_dir = Path(output_dir) / Path(cropped_path).stem
        
        basename = os.path.basename(cropped_path)
        bbox_info = bbox_info_dict.get(basename, None)
        
        if bbox_info:
            continue
        else:
            logger.warning(f"No BBox info - using Affine only")
        
        try:
            results = experiment.run_experiment(
                cropped_nifti_path=cropped_path,
                output_dir=file_output_dir,
                original_nifti_path=original_path,
                cropping_time_info=0,
                quality_levels=quality_levels,
                bbox_info=bbox_info
            )
            
            if results:
                experiment.create_report(results, file_output_dir)
                successful += 1
                
        except Exception as e:
            logger.error(f"Error: {e}")


def batch_process_original_only(original_dir, output_dir, quality_levels,
                                max_files=None, wavelet='bior4.4', level=3):
    """Batch process original files without ROI"""
    original_files = sorted(glob.glob(os.path.join(original_dir, '*.nii*')))
    
    if not original_files:
        logger.error(f"No NIfTI files found in {original_dir}")
        return
    
    if max_files:
        original_files = original_files[:max_files]
    
    
    experiment = Wavelet3DExperiment(wavelet=wavelet, level=level)
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
                quality_levels=quality_levels,
                bbox_info=None
            )
            
            if results:
                experiment.create_report(results, file_output_dir)
                successful += 1
                
        except Exception as e:
            logger.error(f"Error: {e}")
    


# ========== Command Line Interface ==========

def main():
    parser = argparse.ArgumentParser(
        description='3D Wavelet Medical Image Compression Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ROI mode
  python jpeg_3d.py \\
      --cropped_dir ./cropped_volumes \\
      --original_dir ./nifti \\
      --output ./results \\
      --bbox_file ./bbox_info.csv

  # Full mode
  python jpeg_3d.py \\
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
    parser.add_argument('--bbox_file', help='CSV/JSON with BBox info')
    parser.add_argument('--quality_levels', nargs='+', type=int, default=[1, 3, 5],
                       choices=[1,2,3,4,5], help='Quality levels')
    parser.add_argument('--max_files', type=int, help='Maximum files to process')
    parser.add_argument('--use_original_only', action='store_true',
                       help='Process original files without ROI')
    parser.add_argument('--wavelet', default='bior4.4',
                       help='Wavelet type (default: bior4.4)')
    parser.add_argument('--level', type=int, default=3,
                       help='Decomposition level (default: 3)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.use_original_only and args.original_dir:
        batch_process_original_only(
            original_dir=args.original_dir,
            output_dir=args.output,
            quality_levels=args.quality_levels,
            max_files=args.max_files,
            wavelet=args.wavelet,
            level=args.level
        )
    
    elif args.cropped:
        experiment = Wavelet3DExperiment(wavelet=args.wavelet, level=args.level)
        results = experiment.run_experiment(
            cropped_nifti_path=args.cropped,
            output_dir=args.output,
            original_nifti_path=args.original,
            quality_levels=args.quality_levels,
            bbox_info=None
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
            bbox_file=args.bbox_file,
            max_files=args.max_files,
            wavelet=args.wavelet,
            level=args.level
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("3D Wavelet Medical Image Compression Evaluation")
    logger.info("="*70)
    main()