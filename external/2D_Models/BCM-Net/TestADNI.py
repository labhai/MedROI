import argparse
import json
import os
import tempfile
import time
import numpy as np
import torch
import nibabel as nib
import pandas as pd
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import re
import cv2

from Modules.Utils import (get_ref_idx, write_uintx, write_bytes, 
                           read_uintx, read_bytes, calculate_decompression_order)
from Network import Network

torch.backends.cudnn.deterministic = True
NUM_LOSSLESS_BITSTREAMS = 4


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
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'bbox': self.bbox,
            'original_shape': list(self.original_shape) if self.original_shape else None,
            'affine_matrix': self.affine_matrix.tolist() if self.affine_matrix is not None else None
        }
    
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


# ========== Quality Evaluation ==========

class CompressionEvaluator:
    """Compression quality and efficiency evaluator"""
    
    @staticmethod
    def calculate_metrics(original, reconstructed):
        """
        Calculate image quality metrics.
        
        Args:
            original: original image array
            reconstructed: reconstructed image array
            
        Returns:
            mse: Mean Squared Error
            psnr: Peak Signal-to-Noise Ratio (dB)
            ssim_val: Structural Similarity Index
        """
        # MSE
        mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
        
        # PSNR
        if mse == 0:
            psnr = 100.0
        elif mse < 1e-10:
            psnr = 20 * np.log10(255 / np.sqrt(mse))
        else:
            psnr = 20 * np.log10(255 / np.sqrt(mse))
        
        # SSIM
        if original.ndim == 3:  # 3D volume
            ssim_vals = []
            for i in range(original.shape[2]):
                s = ssim(original[:, :, i], reconstructed[:, :, i], data_range=255)
                ssim_vals.append(s)
            ssim_val = np.mean(ssim_vals)
        else:  # 2D slice
            ssim_val = ssim(original, reconstructed, data_range=255)
        
        return mse, psnr, ssim_val
    
    @staticmethod
    def calculate_compression_metrics(original_volume, compressed_size, 
                                    cropped_volume=None, is_roi=False, 
                                    roi_metadata=None):
        """
        Calculate compression efficiency metrics.
        
        Args:
            original_volume: original full volume
            compressed_size: compressed size in bytes
            cropped_volume: cropped ROI volume (for ROI mode)
            is_roi: whether this is ROI data
            roi_metadata: ROIMetadata object
            
        Returns:
            dict: compression metrics
        """
        metadata_size = ROIMetadata.TOTAL_SIZE if roi_metadata is not None else 0
        total_compressed_size = compressed_size + metadata_size
        
        if is_roi and cropped_volume is not None:
            # ROI mode: compare with original full volume
            original_size = original_volume.size * original_volume.itemsize
            cropped_size = cropped_volume.size * cropped_volume.itemsize
            total_pixels = original_volume.size
            
            compression_ratio = original_size / total_compressed_size
            bpp = (total_compressed_size * 8) / total_pixels
            
            print(f"\n[COMPRESSION METRICS - ROI]")
            print(f"  Original size: {original_size:,} bytes")
            print(f"  Compressed size: {compressed_size:,} bytes")
            print(f"  Metadata size: {metadata_size} bytes")
            print(f"  Total compressed: {total_compressed_size:,} bytes")
            print(f"  Compression ratio: {compression_ratio:.2f}:1")
            print(f"  BPP: {bpp:.4f}")
            
            return {
                'compression_ratio': compression_ratio,
                'bpp': bpp,
                'original_full_size': original_size,
                'cropped_size': cropped_size,
                'compressed_size': compressed_size,
                'metadata_size': metadata_size,
                'total_compressed_size': total_compressed_size
            }
        else:
            # Full volume mode
            original_size = original_volume.size * original_volume.itemsize
            total_pixels = original_volume.size
            
            compression_ratio = original_size / compressed_size
            bpp = (compressed_size * 8) / total_pixels
            
            print(f"\n[COMPRESSION METRICS - FULL]")
            print(f"  Original size: {original_size:,} bytes")
            print(f"  Compressed size: {compressed_size:,} bytes")
            print(f"  Compression ratio: {compression_ratio:.2f}:1")
            print(f"  BPP: {bpp:.4f}")
            
            return {
                'compression_ratio': compression_ratio,
                'bpp': bpp,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'metadata_size': 0,
                'total_compressed_size': compressed_size
            }


# ========== Utility Functions ==========

def load_bbox_info(bbox_csv_path):
    """Load 3D bounding box information from CSV"""
    if not bbox_csv_path or not os.path.exists(bbox_csv_path):
        print(f"Warning: bbox CSV not found: {bbox_csv_path}")
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
        
        print(f"Loaded bbox info for {len(bbox_dict)} subjects")
        return bbox_dict
        
    except Exception as e:
        print(f"Failed to load bbox CSV: {e}")
        return {}


def load_nifti_as_yxz(path: str) -> tuple:
    """
    Load NIfTI file and transpose to (Y, X, Z) format.
    
    Args:
        path: path to NIfTI file
        
    Returns:
        volume: numpy array (Y, X, Z)
        affine: 4x4 affine matrix
    """
    nii = nib.load(path)
    vol = np.array(nii.dataobj)
    if vol.ndim == 4:
        vol = vol[:, :, :, 0]
    # NIfTI (X,Y,Z) -> (Y,X,Z)
    vol = np.transpose(vol, (1, 0, 2)).astype(np.int32)
    return vol, nii.affine


def numeric_sort_key(filepath):
    """Extract numeric values from filename for sorting"""
    filename = os.path.basename(filepath)
    nums = re.findall(r'\d+', filename)
    return list(map(int, nums))


def is_nifti(p: str) -> bool:
    """Check if file is NIfTI format"""
    p = p.lower()
    return p.endswith(".nii") or p.endswith(".nii.gz")


def pad_roi_slice_to_full(patch, roi_metadata):
    """
    Pad ROI slice to original full size.
    
    Args:
        patch: ROI slice (H_roi, W_roi)
        roi_metadata: ROIMetadata object
        
    Returns:
        canvas: padded full-size slice (H_full, W_full)
    """
    x0 = roi_metadata.bbox["x_min"]
    x1 = roi_metadata.bbox["x_max"]
    y0 = roi_metadata.bbox["y_min"]
    y1 = roi_metadata.bbox["y_max"]

    Y, X, Z = roi_metadata.original_shape
    canvas = np.zeros((Y, X), dtype=patch.dtype)

    expected_h = y1 - y0 + 1
    expected_w = x1 - x0 + 1

    if patch.shape != (expected_h, expected_w):
        raise ValueError(
            f"Patch shape mismatch: patch={patch.shape}, expected=({expected_h}, {expected_w})"
        )

    canvas[y0:y1+1, x0:x1+1] = patch
    return canvas


def get_target_slice_index(volume_shape, roi_metadata=None, target_anatomical_slice=None):
    """
    Get slice index for same anatomical position in ROI and Original volumes.
    
    Args:
        volume_shape: (H, W, D) current volume shape
        roi_metadata: ROIMetadata object (for ROI)
        target_anatomical_slice: target slice number in original volume
        
    Returns:
        slice_idx: slice index to extract from current volume
        anatomical_position: anatomical position in original volume
    """
    depth = volume_shape[2]
    
    if roi_metadata is None or roi_metadata.bbox is None:
        # Original data
        if target_anatomical_slice is None:
            slice_idx = depth // 2
        else:
            slice_idx = target_anatomical_slice
        anatomical_position = slice_idx
        
    else:
        # ROI data
        z_min = roi_metadata.bbox['z_min']
        z_max = roi_metadata.bbox['z_max']
        
        if target_anatomical_slice is None:
            slice_idx = (z_max - z_min) // 2
            anatomical_position = z_min + slice_idx
        else:
            if target_anatomical_slice < z_min or target_anatomical_slice >= z_max:
                print(f"   Warning: Target slice {target_anatomical_slice} outside ROI range [{z_min}, {z_max})")
                slice_idx = depth // 2
                anatomical_position = z_min + slice_idx
            else:
                slice_idx = target_anatomical_slice - z_min
                anatomical_position = target_anatomical_slice
    
    print(f"   Slice selection: index={slice_idx}, anatomical_position={anatomical_position}")
    return slice_idx, anatomical_position


# ========== VVC Compression Functions ==========

def lossy_compress(lossy_encoder_path: str, lossy_cfg_path: list, yuv_filepath: str, 
                   bin_filepath: str, rec_filepath: str, out_txt_filepath: str, 
                   num_slices: int, height: int, width: int, qp: int) -> None:
    """
    Lossy compress slices using VVC encoder.
    
    Note: This implementation bypasses config files and passes all parameters via command line.
    """
    enc_cmd = (
        f"{lossy_encoder_path}"
        f" --InputFile={yuv_filepath}"
        f" --BitstreamFile={bin_filepath}"
        f" --ReconFile={rec_filepath}"
        f" --SourceWidth={width}"
        f" --SourceHeight={height}"
        f" --InputBitDepth=16"
        f" --InternalBitDepth=16"
        f" --MaxBitDepthConstraint=16"
        f" --OutputBitDepth=16"
        f" --InputChromaFormat=400"
        f" --FrameRate=30"
        f" --FramesToBeEncoded={num_slices}"
        f" --IntraPeriod=1"
        f" --DecodingRefreshType=1"
        f" --GOPSize=1"
        f" --Level=6.1"
        f" --QP={qp}"
        f" > {out_txt_filepath} 2>&1"
    )
    
    print(f"[VVC] Encoding {num_slices} frames ({height}x{width}), QP={qp}")
    ret = os.system(enc_cmd)
    
    if ret != 0:
        raise RuntimeError("VVC encoding failed")


def lossy_decompress(lossy_decoder_path: str, bin_filepath: str, 
                     rec_filepath: str, out_txt_filepath: str) -> None:
    """Decompress slices using VVC decoder"""
    dec_cmd = f'{lossy_decoder_path} -b {bin_filepath} -o {rec_filepath} -d 16 > {out_txt_filepath}'
    os.system(dec_cmd)


# ========== BCM-Net Compression Functions ==========

@torch.no_grad()
def lossless_compress(net: Network, ori_slices: list, rec_slices: list) -> tuple:
    """
    Lossless compress residues using BCM-Net.
    
    Args:
        net: BCM-Net network
        ori_slices: original slices
        rec_slices: VVC reconstructed slices
        
    Returns:
        bitstreams: compressed bitstreams
        residues_min: minimum residue value
        residues_max: maximum residue value
    """
    assert len(ori_slices) == len(rec_slices)
    num_slices = len(ori_slices)
    bitstreams = []
    residues_min = 1e9
    residues_max = -1e8

    # Find residue range
    for slice_idx in range(num_slices):
        ori_slice = torch.from_numpy(ori_slices[slice_idx].astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0)
        rec_slice = torch.from_numpy(rec_slices[slice_idx].astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0)
        residues = ori_slice - rec_slice
        residues_min = min(residues_min, int(residues.min()))
        residues_max = max(residues_max, int(residues.max()))
    
    # Compress residues
    for slice_idx in range(num_slices):
        ori_slice = torch.from_numpy(ori_slices[slice_idx].astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(next(net.parameters()).device)
        rec_slice = torch.from_numpy(rec_slices[slice_idx].astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(next(net.parameters()).device)

        forward_ref_idx, backward_ref_idx = get_ref_idx(slice_idx=slice_idx, num_slices=num_slices)
        ref_forward = None if forward_ref_idx == -1 else \
            torch.from_numpy(ori_slices[forward_ref_idx].astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(next(net.parameters()).device)
        ref_backward = None if backward_ref_idx == -1 else \
            torch.from_numpy(ori_slices[backward_ref_idx].astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(next(net.parameters()).device)

        residues = ori_slice - rec_slice
        bitstreams.append(net.compress(residues=residues, x_tilde=rec_slice, x_min=residues_min, 
                                      x_max=residues_max, ref_forward=ref_forward, ref_backward=ref_backward))

    return bitstreams, residues_min, residues_max


@torch.no_grad()
def lossless_decompress(net: Network, bitstreams: list, rec_slices: list, 
                        residues_min: int, residues_max: int) -> np.ndarray:
    """
    Decompress residues using BCM-Net.
    
    Args:
        net: BCM-Net network
        bitstreams: compressed bitstreams
        rec_slices: VVC reconstructed slices
        residues_min: minimum residue value
        residues_max: maximum residue value
        
    Returns:
        decoded_slices: decompressed volume
    """
    assert len(bitstreams) == len(rec_slices)
    num_slices = len(bitstreams)
    decoded_slices = [torch.zeros(1), ] * num_slices
    slices_indices = calculate_decompression_order(num_slices)
    
    for slice_idx in slices_indices:
        rec_slice = torch.from_numpy(rec_slices[slice_idx].astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(next(net.parameters()).device)

        forward_ref_idx, backward_ref_idx = get_ref_idx(slice_idx=slice_idx, num_slices=num_slices)
        ref_forward = None if forward_ref_idx == -1 else decoded_slices[forward_ref_idx]
        ref_backward = None if backward_ref_idx == -1 else decoded_slices[backward_ref_idx]
        
        residues = net.decompress(strings=bitstreams[slice_idx], x_tilde=rec_slice, 
                                 x_min=residues_min, x_max=residues_max, 
                                 ref_forward=ref_forward, ref_backward=ref_backward)

        decoded_slices[slice_idx] = residues + rec_slice

    return torch.stack(decoded_slices, dim=-1).squeeze().cpu().numpy().astype(np.int32)


# ========== Bitstream Management ==========

def merge_bitstreams(lossless_bitstreams: list, lossy_bin_filepath: str, 
                    dst_bin_filepath: str, num_slices: int, height: int, width: int, 
                    minima: int, residues_min: int, residues_max: int, 
                    scale_factor: float, maxima: int, original_height: int, 
                    original_width: int, roi_metadata: ROIMetadata = None):
    """Merge header information, lossy and lossless bitstreams"""
    with open(dst_bin_filepath, mode='wb') as f:
        write_uintx(f, value=abs(minima), x=16)
        write_uintx(f, value=abs(residues_min), x=16)
        write_uintx(f, value=residues_max, x=16)
        write_uintx(f, value=num_slices, x=16)
        write_uintx(f, value=width, x=16)
        write_uintx(f, value=height, x=16)
        
        scale_factor_int = int(scale_factor * 1000000)
        write_uintx(f, value=scale_factor_int, x=32)
        write_uintx(f, value=maxima, x=32)
        
        write_uintx(f, value=original_height, x=16)
        write_uintx(f, value=original_width, x=16)
        
        # ROI metadata flag and data
        has_roi_metadata = 1 if roi_metadata is not None else 0
        write_uintx(f, value=has_roi_metadata, x=8)
        
        if roi_metadata is not None:
            metadata_bytes = roi_metadata.to_bytes()
            f.write(metadata_bytes)
            print(f"  [METADATA] Saved ROI metadata: {len(metadata_bytes)} bytes")
        
        len_lossy_bitstream = os.path.getsize(lossy_bin_filepath)
        write_uintx(f, value=len_lossy_bitstream, x=32)
        
        with open(lossy_bin_filepath, mode='rb') as src_f:
            data = src_f.read()
            f.write(data)
        
        for bitstream in lossless_bitstreams:
            for i in range(NUM_LOSSLESS_BITSTREAMS):
                length = len(bitstream[i])
                write_uintx(f, value=length, x=32)
                write_bytes(f, values=bitstream[i])


def parse_bitstreams(bitstream_filepath: str, lossy_bin_filepath: str) -> tuple:
    """Parse bitstreams and split to lossy and lossless bitstreams"""
    with open(bitstream_filepath, mode='rb') as f:
        abs_minima = read_uintx(f, x=16)
        minima = -abs_minima
        abs_residues_min = read_uintx(f, x=16)
        residues_min = -abs_residues_min
        residues_max = read_uintx(f, x=16)
        num_slices = read_uintx(f, x=16)
        width = read_uintx(f, x=16)
        height = read_uintx(f, x=16)
        
        scale_factor_int = read_uintx(f, x=32)
        scale_factor = scale_factor_int / 1000000.0
        maxima = read_uintx(f, x=32)
        
        original_height = read_uintx(f, x=16)
        original_width = read_uintx(f, x=16)
        
        # ROI metadata
        has_roi_metadata = read_uintx(f, x=8)
        roi_metadata = None
        
        if has_roi_metadata == 1:
            metadata_bytes = f.read(ROIMetadata.TOTAL_SIZE)
            roi_metadata = ROIMetadata.from_bytes(metadata_bytes)
            print(f"  [METADATA] Loaded ROI metadata")
        
        len_lossy_bitstream = read_uintx(f, x=32)
        with open(lossy_bin_filepath, mode='wb') as lossy_f:
            data = f.read(len_lossy_bitstream)
            lossy_f.write(data)
        
        lossless_bitstreams = []
        for i in range(num_slices):
            bitstream = []
            for j in range(NUM_LOSSLESS_BITSTREAMS):
                length = read_uintx(f, x=32)
                bitstream.append(read_bytes(f, n=length))
            lossless_bitstreams.append(bitstream)
    
    return (lossless_bitstreams, num_slices, height, width, minima, residues_min, 
            residues_max, scale_factor, maxima, original_height, original_width, roi_metadata)


# ========== Main Compression/Decompression Functions ==========

@torch.no_grad()
def compress_enhanced(nii_filepath: str, bin_filepath: str, lossless_net: Network,
                     lossy_cfg_path: list, lossy_encoder_path: str, 
                     lossy_decoder_path: str, qp: int, 
                     original_nii_filepath: str = None, is_roi: bool = False,
                     bbox_info: dict = None) -> dict:
    """
    Compress NIfTI volume with two-stage approach (VVC + BCM-Net).
    
    Args:
        nii_filepath: path to NIfTI file
        bin_filepath: output bitstream path
        lossless_net: BCM-Net network
        lossy_cfg_path: VVC config paths (for compatibility, not used)
        lossy_encoder_path: VVC encoder path
        lossy_decoder_path: VVC decoder path
        qp: quantization parameter
        original_nii_filepath: original file path (for ROI mode)
        is_roi: whether this is ROI data
        bbox_info: bounding box information
        
    Returns:
        dict: compression metrics
    """
    torch.cuda.synchronize()
    total_start_time = time.time()
    
    # Load volume as (Y, X, Z)
    volume_data, affine_matrix = load_nifti_as_yxz(nii_filepath)
    height, width, num_slices = volume_data.shape

    # Pad to even dimensions for BCM-Net
    original_height, original_width = height, width
    pad_height = (height % 2)
    pad_width = (width % 2)

    if pad_height or pad_width:
        print(f"\n[PADDING] {height}x{width} -> ", end='')
        volume_data = np.pad(volume_data, 
                            ((0, pad_height), (0, pad_width), (0, 0)),
                            mode='edge')
        height, width, num_slices = volume_data.shape
        print(f"{height}x{width}")
    
    # Load original volume for ROI mode
    original_volume = None
    if is_roi and original_nii_filepath:
        original_volume, _ = load_nifti_as_yxz(original_nii_filepath)
    
    # Create ROI metadata
    roi_metadata = None
    if is_roi and bbox_info and original_volume is not None and affine_matrix is not None:
        bbox_only = {k: int(bbox_info[k]) for k in ['x_min','x_max','y_min','y_max','z_min','z_max']}
        roi_metadata = ROIMetadata(
            bbox=bbox_only,
            original_shape=original_volume.shape,
            affine_matrix=affine_matrix
        )
        print(f"ROI Metadata created")
    
    # Shift to [0, max]
    minima = volume_data.min()
    volume_data = volume_data - minima
    maxima = volume_data.max()
    
    # Scale to 16-bit range if needed
    if maxima > 65535:
        print(f"   Scaling data to 16-bit range (max={maxima})")
        scale_factor = maxima / 65535.0
        volume_data = (volume_data / scale_factor).astype(np.uint16)
    else:
        scale_factor = 1.0
        volume_data = volume_data.astype(np.uint16)
    
    slices = [volume_data[:, :, i] for i in range(num_slices)]
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write to YUV
        yuv_filepath = os.path.join(tmp_dir, 'input.yuv')
        with open(yuv_filepath, mode='wb') as f:
            for s in slices:
                np.asarray(s, dtype=np.uint16).tofile(f)
        
        lossy_bin_filepath = os.path.join(tmp_dir, 'lossy.bin')
        lossy_rec_filepath = os.path.join(tmp_dir, 'rec.yuv')
        filename = os.path.basename(nii_filepath)
        lossy_out_txt_filepath = os.path.join(
            os.path.dirname(bin_filepath),
            f"{os.path.splitext(filename)[0]}_encode.log"
        )
        
        # Lossy compression
        lossy_start_time = time.time()
        lossy_compress(yuv_filepath=yuv_filepath, bin_filepath=lossy_bin_filepath, 
                      rec_filepath=lossy_rec_filepath, lossy_cfg_path=lossy_cfg_path, 
                      lossy_encoder_path=lossy_encoder_path, 
                      out_txt_filepath=lossy_out_txt_filepath,
                      num_slices=num_slices, height=height, width=width, qp=qp)
        
        # Lossy decompression
        lossy_decompress(bin_filepath=lossy_bin_filepath, rec_filepath=lossy_rec_filepath, 
                        lossy_decoder_path=lossy_decoder_path, 
                        out_txt_filepath=lossy_out_txt_filepath)
        
        torch.cuda.synchronize()
        lossy_runtime = time.time() - lossy_start_time
        
        # Lossless compression
        torch.cuda.synchronize()
        lossless_start_time = time.time()
        
        with open(yuv_filepath, mode='rb') as f:
            ori_slices = [np.reshape(np.frombuffer(f.read(2 * height * width), dtype=np.uint16), 
                                    (height, width)).astype(np.uint16) for _ in range(num_slices)]
        with open(lossy_rec_filepath, mode='rb') as f:
            rec_slices = [np.reshape(np.frombuffer(f.read(2 * height * width), dtype=np.uint16), 
                                    (height, width)).astype(np.uint16) for _ in range(num_slices)]
        
        lossless_bitstreams, residues_min, residues_max = lossless_compress(
            net=lossless_net, ori_slices=ori_slices, rec_slices=rec_slices
        )
        
        merge_bitstreams(lossy_bin_filepath=lossy_bin_filepath, dst_bin_filepath=bin_filepath, 
                        lossless_bitstreams=lossless_bitstreams, num_slices=num_slices, 
                        height=height, width=width, minima=minima, 
                        residues_min=residues_min, residues_max=residues_max,
                        scale_factor=scale_factor, maxima=maxima,
                        original_height=original_height, original_width=original_width,
                        roi_metadata=roi_metadata)
        
        torch.cuda.synchronize()
        lossless_runtime = time.time() - lossless_start_time
        
        # Calculate compression metrics
        compressed_size = os.path.getsize(bin_filepath)
        lossy_size = os.path.getsize(lossy_bin_filepath)
        lossless_size = compressed_size - lossy_size
        
        evaluator = CompressionEvaluator()
        if is_roi and original_volume is not None:
            comp_metrics = evaluator.calculate_compression_metrics(
                original_volume=original_volume,
                compressed_size=compressed_size,
                cropped_volume=volume_data,
                is_roi=True,
                roi_metadata=roi_metadata
            )
        else:
            comp_metrics = evaluator.calculate_compression_metrics(
                original_volume=volume_data,
                compressed_size=compressed_size,
                is_roi=False,
                roi_metadata=None
            )
        
        torch.cuda.synchronize()
        total_runtime = time.time() - total_start_time

        return {
            **comp_metrics,
            'lossy_size': lossy_size,
            'lossless_size': lossless_size,
            'lossy_bpp': (lossy_size * 8) / (height * width * num_slices),
            'lossless_bpp': (lossless_size * 8) / (height * width * num_slices),
            'lossy_runtime': lossy_runtime,
            'lossless_runtime': lossless_runtime,
            'total_runtime': total_runtime
        }


@torch.no_grad()
def decompress_enhanced(bin_filepath: str, decoded_filepath: str, 
                       lossy_decoder_path: str, lossless_net: Network, 
                       subject_id: str = None,
                       data_type: str = 'original',
                       target_anatomical_slice: int = None) -> dict:
    """
    Decompress bitstream to NIfTI volume.
    
    Args:
        bin_filepath: compressed bitstream path
        decoded_filepath: output numpy file path
        lossy_decoder_path: VVC decoder path
        lossless_net: BCM-Net network
        subject_id: subject ID
        data_type: 'original' or 'roi'
        target_anatomical_slice: target anatomical slice position
        
    Returns:
        dict: decompression metrics
    """
    torch.cuda.synchronize()
    start_time = time.time()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        lossy_bin_filepath = os.path.join(tmp_dir, 'lossy.bin')
        
        # Parse bitstreams
        result = parse_bitstreams(bitstream_filepath=bin_filepath, 
                                 lossy_bin_filepath=lossy_bin_filepath)
        (lossless_bitstreams, num_slices, height, width, minima, residues_min, 
         residues_max, scale_factor, maxima, original_height, original_width, 
         roi_metadata) = result
                
        # Lossy decompression
        rec_filepath = os.path.join(tmp_dir, 'rec.yuv')
        lossy_out_txt_filepath = os.path.join(tmp_dir, 'out.txt')
        lossy_decompress(bin_filepath=lossy_bin_filepath, rec_filepath=rec_filepath, 
                        lossy_decoder_path=lossy_decoder_path, 
                        out_txt_filepath=lossy_out_txt_filepath)
        
        # Lossless decompression
        with open(rec_filepath, mode='rb') as f:
            rec_slices = [np.reshape(np.frombuffer(f.read(2 * height * width), 
                                                   dtype=np.uint16), 
                                    (height, width)).astype(np.uint16) 
                         for _ in range(num_slices)]
        
        decoded_slices = lossless_decompress(net=lossless_net, 
                                            bitstreams=lossless_bitstreams, 
                                            rec_slices=rec_slices, 
                                            residues_min=residues_min, 
                                            residues_max=residues_max)
        
        # Unscale
        if scale_factor != 1.0:
            decoded_slices = (decoded_slices.astype(np.float64) * scale_factor).astype(np.int32)
        
        # Shift back
        decoded_slices = decoded_slices + minima
        
        # Unpad
        if height != original_height or width != original_width:
            decoded_slices = decoded_slices[:original_height, :original_width, :]

        # Save numpy array
        np.save(decoded_filepath, decoded_slices)
        
        torch.cuda.synchronize()
        
        return {
            'decompression_time': time.time() - start_time,
            'roi_metadata': roi_metadata
        }


# ========== Main Tester Class ==========

class EnhancedTester:
    """Main tester class for compression evaluation"""
    
    def __init__(self):
        self.args = self.parse_args()
        self.net = Network(bit_depth=self.args.bit_depth)
        self.net = self.net.to("cuda" if self.args.gpu else "cpu").eval()
        self.load_weights()
        self.evaluator = CompressionEvaluator()
        
        # Load bbox information
        self.bbox_dict = {}
        if self.args.is_roi and self.args.bbox_csv:
            self.bbox_dict = load_bbox_info(self.args.bbox_csv)
        
        # Prepare file list
        self.data_files = [os.path.join(self.args.data_root, f) 
                          for f in os.listdir(self.args.data_root) 
                          if is_nifti(f)]
        
        self.data_files.sort(key=numeric_sort_key)
        
        if self.args.test_file:
            self.data_files = [f for f in self.data_files 
                              if os.path.basename(f) == self.args.test_file]
            if not self.data_files:
                raise ValueError(f"File {self.args.test_file} not found!")
            print(f"Testing only: {self.args.test_file}")
        
        if self.args.max_files:
            self.data_files = self.data_files[:self.args.max_files]
            print(f"Testing only first {self.args.max_files} files")
    
    @torch.no_grad()
    def test(self):
        """Run compression evaluation on all files"""
        os.makedirs(self.args.save_directory, exist_ok=True)

        # Determine target anatomical slice
        target_anatomical_slice = None
        if self.data_files:
            if self.args.is_roi and self.args.original_root:
                candidates = [
                    os.path.join(self.args.original_root, f)
                    for f in os.listdir(self.args.original_root)
                    if is_nifti(f)
                ]
                if not candidates:
                    raise RuntimeError(f"No NIfTI files found in original_root: {self.args.original_root}")
                candidates.sort(key=numeric_sort_key)
                first_file_path = candidates[0]
            else:
                first_file_path = self.data_files[0]

            first_volume, _ = load_nifti_as_yxz(first_file_path)
            target_anatomical_slice = first_volume.shape[2] // 2
            print(f"\n{'='*60}")
            print(f"TARGET SLICE: {target_anatomical_slice} (based on first subject)")
            print(f"{'='*60}")
        
        results_list = []
        
        for nii_filepath in self.data_files:
            filename = os.path.basename(nii_filepath)
            print(f"\n{'='*60}")
            print(f"Processing: {filename}")
            print(f"{'='*60}")
            
            bin_filepath = os.path.join(self.args.save_directory, 
                                       os.path.splitext(filename)[0] + ".bin")
            decoded_filepath = os.path.join(self.args.save_directory, 
                                           os.path.splitext(filename)[0] + ".npy")
            
            # Extract subject ID and find bbox
            subject_id = filename.replace('.nii.gz', '').replace('.nii', '')
            bbox_info = self.bbox_dict.get(subject_id) if self.args.is_roi else None
            
            # Find original file path for ROI mode
            original_nii_filepath = None
            if self.args.is_roi and self.args.original_root:
                original_filename = filename.replace('_cropped', '').replace('_roi', '')
                if original_filename.endswith('.nii.gz'):
                    original_filename = original_filename[:-3]
                
                original_nii_filepath = os.path.join(self.args.original_root, original_filename)
                
                if not os.path.exists(original_nii_filepath):
                    print(f"Warning: Original file not found: {original_nii_filepath}")
                    original_nii_filepath = None
                else:
                    print(f"Found original: {original_filename}")
            
            # Compression
            compress_metrics = compress_enhanced(
                nii_filepath=nii_filepath,
                bin_filepath=bin_filepath,
                lossless_net=self.net,
                qp=self.args.qp,
                lossy_cfg_path=self.args.lossy_cfg_path,
                lossy_encoder_path=self.args.lossy_encoder_path,
                lossy_decoder_path=self.args.lossy_decoder_path,
                original_nii_filepath=original_nii_filepath,
                is_roi=self.args.is_roi,
                bbox_info=bbox_info
            )
            
            # Decompression 
            decompress_metrics = decompress_enhanced(
                bin_filepath=bin_filepath,
                decoded_filepath=decoded_filepath,
                lossy_decoder_path=self.args.lossy_decoder_path,
                lossless_net=self.net,
                subject_id=subject_id,
                data_type=data_type,
                target_anatomical_slice=anatomical_pos
            )
            
            # Quality verification
            decoded_volume = np.load(decoded_filepath).astype(np.int32)
            ori_volume_data, _ = load_nifti_as_yxz(nii_filepath)
            
            diff = np.abs(ori_volume_data - decoded_volume)
            max_diff = diff.max()
            mean_diff = diff.mean()

            original_range = ori_volume_data.max() - ori_volume_data.min()
            bits_required = int(np.ceil(np.log2(original_range + 1))) if original_range > 0 else 1

            if bits_required <= 16:
                acceptable_error = 1
            else:
                scale_factor_estimate = original_range / 65535.0
                acceptable_error = int(scale_factor_estimate * 2)
                
            print(f"\n[LOSSLESS VERIFICATION]")
            print(f"   Max error: {max_diff} (acceptable: ≤{acceptable_error})")
            print(f"   Mean error: {mean_diff:.6f}")

            if max_diff == 0:
                print(f"   Perfect lossless")
            elif max_diff <= acceptable_error:
                print(f"   Near-lossless (acceptable)")
            else:
                print(f"   Error too large")

            assert max_diff <= acceptable_error, \
                f'Verification failed: {filename}'
                
            # Calculate quality metrics
            ori_min, ori_max = ori_volume_data.min(), ori_volume_data.max()
            dec_min, dec_max = decoded_volume.min(), decoded_volume.max()

            value_range = max(ori_max, dec_max) - min(ori_min, dec_min)
            ori_norm = ((ori_volume_data - min(ori_min, dec_min)) / value_range * 255).astype(np.uint8)
            dec_norm = ((decoded_volume - min(ori_min, dec_min)) / value_range * 255).astype(np.uint8)

            mse, psnr, ssim_val = self.evaluator.calculate_metrics(ori_norm, dec_norm)

            # Store results
            result = {
                'filename': filename,
                'subject_id': subject_id,
                'is_roi': self.args.is_roi,
                'has_bbox': bbox_info is not None,
                'anatomical_slice': anatomical_pos,
                'bits_required': bits_required,
                'max_error': int(max_diff),
                'mean_error': float(mean_diff),
                'acceptable_error': int(acceptable_error),
                'mse': mse,
                'psnr': psnr,
                'ssim': ssim_val,
                **compress_metrics,
                **decompress_metrics,
                'qp': self.args.qp
            }

            results_list.append(result)

            print(f"\n{filename}")
            print(f"   BPP: {result['bpp']:.4f}")
            print(f"   Compression Ratio: {result['compression_ratio']:.2f}:1")
            print(f"   PSNR: {psnr:.2f} dB")
            print(f"   SSIM: {ssim_val:.4f}")

        # Calculate averages and create reports
        avg_results = {
            'avg_bpp': np.mean([r['bpp'] for r in results_list]),
            'avg_compression_ratio': np.mean([r['compression_ratio'] for r in results_list]),
            'avg_psnr': np.mean([r['psnr'] for r in results_list]),
            'avg_ssim': np.mean([r['ssim'] for r in results_list]),
            'avg_total_time': np.mean([r['total_runtime'] for r in results_list]),
            'avg_metadata_size': np.mean([r.get('metadata_size', 0) for r in results_list])
        }
        
        self.create_reports(results_list, avg_results)
        
    def create_reports(self, results_list, avg_results):
        """Generate CSV and JSON reports"""
        
        # Convert ROIMetadata to dict for JSON serialization
        serializable_results = []
        for result in results_list:
            result_copy = result.copy()
            if 'roi_metadata' in result_copy and isinstance(result_copy['roi_metadata'], ROIMetadata):
                result_copy['roi_metadata'] = result_copy['roi_metadata'].to_dict()
            serializable_results.append(result_copy)
        
        # Save JSON
        full_results = {
            'summary': avg_results,
            'details': serializable_results
        }
        json_file = os.path.join(self.args.save_directory, 'results.json')
        with open(json_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        # Save CSV
        df = pd.DataFrame(results_list)
        csv_file = os.path.join(self.args.save_directory, 'results.csv')
        df.to_csv(csv_file, index=False)
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Average BPP: {avg_results['avg_bpp']:.4f}")
        print(f"Average Compression Ratio: {avg_results['avg_compression_ratio']:.2f}:1")
        print(f"Average Metadata Size: {avg_results['avg_metadata_size']:.1f} bytes")
        print(f"Average PSNR: {avg_results['avg_psnr']:.2f} dB")
        print(f"Average SSIM: {avg_results['avg_ssim']:.4f}")
        print(f"Average Time: {avg_results['avg_total_time']:.2f}s")
        print(f"\nResults saved to:")
        print(f"  - {json_file}")
        print(f"  - {csv_file}")
    
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(
            description="BCM-Net Medical Image Compression Evaluation",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Original data compression
  python TestADNI.py \\
      --data_root /path/to/nifti \\
      --save_directory ./results \\
      --qp 0 \\
      --checkpoints ./model.pth

  # ROI data compression
  python TestADNI.py \\
      --data_root /path/to/cropped_volumes \\
      --original_root /path/to/original_nifti \\
      --is_roi \\
      --bbox_csv /path/to/bbox.csv \\
      --save_directory ./results \\
      --qp 0 \\
      --checkpoints ./model.pth
            """
        )
        
        # Required arguments
        parser.add_argument('--data_root', type=str, required=True,
                          help='Path to NIfTI files (ROI or original)')
        parser.add_argument('--save_directory', type=str, required=True,
                          help='Output directory')
        parser.add_argument('--lossy_encoder_path', type=str, required=True,
                          help='Path to VVC encoder')
        parser.add_argument('--lossy_decoder_path', type=str, required=True,
                          help='Path to VVC decoder')
        parser.add_argument('--lossy_cfg_path', type=str, nargs='+', required=True,
                          help='VVC config file paths')
        parser.add_argument('--checkpoints', type=str, required=True,
                          help='Path to BCM-Net checkpoint')
        
        # Optional arguments
        parser.add_argument('--original_root', type=str, default=None,
                          help='Path to original files (required if --is_roi)')
        parser.add_argument('--is_roi', action='store_true',
                          help='Data is ROI (cropped)')
        parser.add_argument('--bbox_csv', type=str, default=None,
                          help='Path to bbox CSV file')
        parser.add_argument('--qp', type=int, default=0,
                          help='Quantization parameter (default: 0)')
        parser.add_argument('--gpu', action='store_true', default=True,
                          help='Use GPU')
        parser.add_argument('--bit_depth', type=int, default=16,
                          help='Bit depth (default: 16)')
        parser.add_argument('--max_files', type=int, default=None,
                          help='Maximum number of files to process')
        parser.add_argument('--test_file', type=str, default=None,
                          help='Test specific file only')
        
        return parser.parse_args()
    
    def load_weights(self):
        """Load model weights"""
        print(f'Loading weights from {self.args.checkpoints}')
        ckpt = torch.load(self.args.checkpoints, 
                         map_location='cuda' if self.args.gpu else 'cpu')
        self.net.load_state_dict(ckpt['network'] if 'network' in ckpt else ckpt)


if __name__ == "__main__":
    tester = EnhancedTester()
    tester.test()