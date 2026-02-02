import torch
import torch.nn.functional as F
from models import TCM_AUXT
from PIL import Image
import nibabel as nib
import warnings
import os
import sys
import math
import argparse
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
import glob

warnings.filterwarnings("ignore")


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
        
        # Affine Matrix (expand 3×3 to 4×4)
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


# ========== NIfTI Processing Functions ==========

def load_nifti_volume(nifti_path):
    """
    Load NIfTI file.
    
    Returns:
        volume: numpy array (H, W, D)
        affine: 4x4 affine matrix
        header: NIfTI header
    """
    try:
        nii = nib.load(nifti_path)
        volume = nii.get_fdata()
        affine = nii.affine
        header = nii.header
        return volume, affine, header
    except Exception as e:
        print(f"Failed to load {nifti_path}: {e}")
        return None, None, None


def normalize_image(image_data):
    """
    Normalize image to [0, 255] range.
    
    Args:
        image_data: 2D numpy array
    Returns:
        normalized: uint8 array in [0, 255]
    """
    min_val, max_val = np.min(image_data), np.max(image_data)
    if max_val > min_val:
        normalized = ((image_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(image_data, dtype=np.uint8)
    return normalized


def normalize_to_tensor(volume_slice, device='cpu'):
    """
    Convert NIfTI slice to tensor for model input.
    
    Args:
        volume_slice: 2D numpy array (H, W)
        device: torch device
    Returns:
        torch.Tensor: (1, 3, H, W) RGB format
    """
    # Normalize to [0, 255]
    normalized = normalize_image(volume_slice)
    
    # Rotate if needed
    normalized = np.rot90(normalized)
    
    # Convert to [0, 1] range
    normalized_float = normalized.astype(np.float32) / 255.0
    
    # Grayscale to RGB (model expects 3 channels)
    rgb_slice = np.stack([normalized_float]*3, axis=0)
    
    # Convert to tensor
    tensor = torch.from_numpy(rgb_slice).float().unsqueeze(0)
    tensor = tensor.to(device)
    
    return tensor


def extract_slices(volume_data, axis=2, max_slices=None):
    """
    Extract 2D slices from 3D volume.
    
    Args:
        volume_data: (H, W, D) numpy array
        axis: slice axis (0=sagittal, 1=coronal, 2=axial)
        max_slices: maximum number of slices to extract (None for all)
    Returns:
        slices: list of 2D numpy arrays
        indices: list of slice indices
    """
    slice_count = volume_data.shape[axis]
    
    if max_slices is None or max_slices >= slice_count:
        indices = range(slice_count)
    else:
        # Sample from middle portion
        step = max(1, slice_count // max_slices)
        indices = range(slice_count // 4, 3 * slice_count // 4, step)[:max_slices]
    
    slices = []
    for i in indices:
        slice_data = np.take(volume_data, i, axis=axis)
        slices.append(slice_data)
    
    return slices, list(indices)


def load_bbox_info(bbox_csv_path):
    """Load 3D bounding box information from CSV"""
    if not os.path.exists(bbox_csv_path):
        print(f"Warning: bbox CSV not found: {bbox_csv_path}")
        return {}
    
    try:
        df = pd.read_csv(bbox_csv_path)
        bbox_dict = {}
        
        for _, row in df.iterrows():
            subject_id = row['subject_id']
            bbox_dict[subject_id] = {
                'x_min': int(row['x_min']), 'x_max': int(row['x_max']),
                'y_min': int(row['y_min']), 'y_max': int(row['y_max']),
                'z_min': int(row['z_min']), 'z_max': int(row['z_max']),
                'original_shape': (int(row['orig_x']), int(row['orig_y']), int(row['orig_z']))
            }
        
        print(f"Loaded bbox info for {len(bbox_dict)} subjects")
        return bbox_dict
        
    except Exception as e:
        print(f"Failed to load bbox CSV: {e}")
        return {}


def load_cropping_times(cropping_log_path):
    """Load cropping time information from CSV or JSON"""
    if not os.path.exists(cropping_log_path):
        return {}
    
    try:
        if cropping_log_path.endswith('.csv'):
            df = pd.read_csv(cropping_log_path)
            return dict(zip(df['filename'], df['total_cropping_time']))
        elif cropping_log_path.endswith('.json'):
            with open(cropping_log_path) as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load cropping times: {e}")
    
    return {}


# ========== Quality Metrics ==========

def compute_psnr(a, b):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = torch.mean((a - b)**2).item()
    if mse == 0:
        return 100
    return -10 * math.log10(mse)


def compute_ssim(a, b):
    """Calculate Structural Similarity Index"""
    from pytorch_msssim import ssim
    return ssim(a, b, data_range=1., size_average=True).item()


def compute_compression_ratio(original_size_bytes, compressed_size_bytes):
    """Calculate compression ratio"""
    if compressed_size_bytes == 0:
        return 0
    return original_size_bytes / compressed_size_bytes


# ========== Image Processing Utilities ==========

def pad(x, p):
    """Pad image to be divisible by p"""
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom), 
                     mode="constant", value=0)
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def crop(x, padding):
    """Remove padding from image"""
    return F.pad(x, (-padding[0], -padding[1], -padding[2], -padding[3]))


def pad_or_crop_to_shape(img_u8, target_hw):
    """
    Center crop/pad image to target shape.
    
    Args:
        img_u8: (H, W) uint8 array
        target_hw: (target_h, target_w) tuple
    Returns:
        img_u8: resized array
    """
    h, w = img_u8.shape
    th, tw = target_hw

    # Center crop if larger
    if h > th:
        top = (h - th) // 2
        img_u8 = img_u8[top:top + th, :]
        h = th
    if w > tw:
        left = (w - tw) // 2
        img_u8 = img_u8[:, left:left + tw]
        w = tw

    # Zero pad if smaller
    pad_top = (th - h) // 2
    pad_bottom = th - h - pad_top
    pad_left = (tw - w) // 2
    pad_right = tw - w - pad_left

    img_u8 = np.pad(img_u8, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                    mode="constant", constant_values=0)
    return img_u8


# ========== Main Processing Function ==========

def process_nifti_file(nifti_path, net, device, args, 
                       bbox_info=None, original_nifti_path=None,
                       cropping_time=0.0, target_slice_idx=None):
    """
    Process NIfTI file slice by slice with compression and quality evaluation.
    
    Args:
        nifti_path: path to NIfTI file
        net: compression model
        device: torch device
        args: command line arguments
        bbox_info: bounding box information (for ROI mode)
        original_nifti_path: path to original volume (for ROI comparison)
        cropping_time: time spent on ROI cropping
        target_slice_idx: specific slice index to process (None for all)
    
    Returns:
        dict: compression results and metrics
    """
    
    # Load NIfTI volume
    volume, affine, header = load_nifti_volume(nifti_path)
    if volume is None:
        return None
    
    H, W, D = volume.shape
    
    # Load original volume for ROI mode
    original_volume = None
    original_affine = affine
    
    if args.is_roi and original_nifti_path and os.path.exists(original_nifti_path):
        original_img = nib.load(original_nifti_path)
        original_volume = original_img.get_fdata()
        original_affine = original_img.affine
    
    # Extract slices
    slices, slice_indices = extract_slices(volume, axis=args.axis, max_slices=None)
    
    # Filter for target slice if specified
    if target_slice_idx is not None and target_slice_idx not in slice_indices:
        return None
    
    # Prepare reconstruction directory
    recon_dir = Path(args.output) / "recon_slices" / Path(nifti_path).stem
    recon_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ROI metadata
    roi_metadata = None
    if args.is_roi and bbox_info:
        roi_metadata = ROIMetadata(
            bbox=bbox_info,
            original_shape=original_volume.shape if original_volume is not None else volume.shape,
            affine_matrix=original_affine
        )
    
    # Initialize metrics
    compressed_slices_data = []
    subject_psnr = []
    subject_ssim = []
    subject_comp_time = 0
    subject_decomp_time = 0
    subject_compressed_size = 0
    processed_slice_count = 0
    
    p = 256  # Padding size for model
    
    # Process each slice
    for i, (cropped_slice, orig_idx) in enumerate(zip(slices, slice_indices)):
        
        # Filter for target slice
        if target_slice_idx is not None and orig_idx != target_slice_idx:
            continue
        
        processed_slice_count += 1
        
        # Convert to tensor
        x = normalize_to_tensor(cropped_slice, device)
        _, _, h_cropped, w_cropped = x.shape
        num_pixels_cropped = h_cropped * w_cropped
        
        # Determine pixel count for BPP calculation
        if args.is_roi and original_volume is not None:
            num_pixels_for_bpp = original_volume.shape[0] * original_volume.shape[1]
        else:
            num_pixels_for_bpp = num_pixels_cropped
        
        # Pad image
        x_padded, padding = pad(x, p)
        
        # Compress and decompress
        with torch.no_grad():
            if args.real:
                # Real compression with entropy coding
                if args.cuda:
                    torch.cuda.synchronize()
                
                comp_start = time.time()
                out_enc = net.compress(x_padded)
                if args.cuda:
                    torch.cuda.synchronize()
                comp_time = time.time() - comp_start
                
                decomp_start = time.time()
                out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
                if args.cuda:
                    torch.cuda.synchronize()
                decomp_time = time.time() - decomp_start
                
                out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
                out_dec["x_hat"].clamp_(0, 1)
                
                compressed_size_bytes = sum(len(s[0]) for s in out_enc["strings"])
                x_hat = out_dec["x_hat"]
                
                compressed_slices_data.append({
                    'slice_idx': int(orig_idx),
                    'compressed_data': out_enc["strings"],
                    'shape': out_enc["shape"],
                    'size_bytes': compressed_size_bytes
                })
                
            else:
                # Forward pass mode (BPP estimation)
                if args.cuda:
                    torch.cuda.synchronize()
                
                comp_start = time.time()
                out_net = net.forward(x_padded)
                if args.cuda:
                    torch.cuda.synchronize()
                total_forward_time = time.time() - comp_start
                
                comp_time = total_forward_time * 0.5
                decomp_time = total_forward_time * 0.5
                
                out_net['x_hat'].clamp_(0, 1)
                out_net["x_hat"] = crop(out_net["x_hat"], padding)
                
                bpp_slice = sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels_for_bpp)
                              for likelihoods in out_net['likelihoods'].values()).item()
                
                compressed_size_bytes = (bpp_slice * num_pixels_for_bpp) / 8.0
                x_hat = out_net["x_hat"]
        
        # Accumulate metrics
        subject_compressed_size += compressed_size_bytes
        subject_comp_time += comp_time
        subject_decomp_time += decomp_time
        
        # Save reconstructed slice
        with torch.no_grad():
            x_hat_img = x_hat.squeeze(0).detach().cpu()
            x_hat_gray = x_hat_img[0]
            roi_u8 = (x_hat_gray.numpy() * 255.0).clip(0, 255).astype(np.uint8)

        if args.is_roi and (original_volume is not None):
            # Match size to original slice
            orig_slice = np.take(original_volume, orig_idx, axis=args.axis)
            orig_slice_rot = np.rot90(orig_slice)
            target_hw = orig_slice_rot.shape
            roi_u8_full = pad_or_crop_to_shape(roi_u8, target_hw)
            out_png = recon_dir / f"slice_{orig_idx:04d}_reconstructed.png"
            Image.fromarray(roi_u8_full).save(out_png)
        else:
            out_png = recon_dir / f"slice_{orig_idx:04d}.png"
            Image.fromarray(roi_u8).save(out_png)
        
        # Calculate quality metrics
        psnr = compute_psnr(x, x_hat)
        ssim_val = compute_ssim(x, x_hat)
        
        subject_psnr.append(psnr)
        subject_ssim.append(ssim_val)
    
    # Return None if no slices processed
    if processed_slice_count == 0:
        return None
    
    # Add metadata size
    metadata_size = ROIMetadata.TOTAL_SIZE if roi_metadata else 0
    total_compressed_with_metadata = subject_compressed_size + metadata_size
    
    # Calculate BPP and compression ratio
    if args.is_roi and original_volume is not None:
        # ROI mode: use original volume dimensions
        total_pixels = (original_volume.shape[0] * original_volume.shape[1] * 
                       processed_slice_count)
        original_size_bytes = total_pixels * 1
    else:
        # Full volume mode: use cropped volume dimensions
        total_pixels = H * W * processed_slice_count
        original_size_bytes = total_pixels * 1
    
    if total_compressed_with_metadata > 0:
        bpp = (total_compressed_with_metadata * 8) / total_pixels
        compression_ratio = original_size_bytes / total_compressed_with_metadata
    else:
        bpp = 0
        compression_ratio = 0
    
    return {
        'volume_shape': (H, W, D),
        'num_slices': processed_slice_count,
        'avg_psnr': np.mean(subject_psnr),
        'avg_ssim': np.mean(subject_ssim),
        'compressed_size': subject_compressed_size,
        'metadata_size': metadata_size,
        'total_compressed_size': total_compressed_with_metadata,
        'bpp': bpp,
        'compression_ratio': compression_ratio,
        'cropping_time': cropping_time,
        'compression_time': subject_comp_time,
        'decompression_time': subject_decomp_time,
        'has_roi_metadata': roi_metadata is not None,
        'roi_metadata': roi_metadata,
        'affine': affine,
        'compressed_slices': compressed_slices_data if args.real else None
    }


# ========== Command Line Interface ==========

def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Medical Image Compression Evaluation for NIfTI volumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ROI compression evaluation
  python eval.py \\
      --cuda --real --is-roi \\
      --bbox-csv ./bbox_info.csv \\
      --checkpoint ./model.pth.tar \\
      --data ./cropped_volumes \\
      --original-dir ./original_volumes \\
      --output ./results

  # Full volume evaluation
  python eval.py \\
      --cuda --real \\
      --checkpoint ./model.pth.tar \\
      --data ./nifti_files \\
      --output ./results
        """
    )
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to NIfTI files directory")
    
    # Optional arguments
    parser.add_argument("--cuda", action="store_true",
                       help="Use CUDA for acceleration")
    parser.add_argument("--real", action="store_true",
                       help="Use real compression with entropy coding")
    parser.add_argument("--output", type=str, default="./results",
                       help="Output directory (default: ./results)")
    
    # ROI-specific arguments
    parser.add_argument("--is-roi", action="store_true",
                       help="Enable ROI compression mode")
    parser.add_argument("--bbox-csv", type=str,
                       help="Path to bounding box CSV file")
    parser.add_argument("--cropping-csv", type=str,
                       help="Path to cropping times CSV")
    parser.add_argument("--original-dir", type=str,
                       help="Directory with original NIfTI files (for ROI comparison)")
    
    # Processing options
    parser.add_argument("--max-subjects", type=int,
                       help="Maximum number of subjects to process")
    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2],
                       help="Slice axis: 0=sagittal, 1=coronal, 2=axial (default: 2)")
    parser.add_argument("--target-slice", type=int,
                       help="Process only specific slice index")
    
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Device setup
    device = 'cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load bbox information
    bbox_dict = {}
    if args.is_roi and args.bbox_csv:
        bbox_dict = load_bbox_info(args.bbox_csv)
    
    # Load cropping times
    cropping_times = {}
    if args.is_roi and args.cropping_csv:
        cropping_times = load_cropping_times(args.cropping_csv)
    
    # Get NIfTI files
    nifti_files = sorted(glob.glob(os.path.join(args.data, '*.nii.gz')) + 
                        glob.glob(os.path.join(args.data, '*.nii')))
    
    if len(nifti_files) == 0:
        print(f"No NIfTI files found in {args.data}")
        return
    
    if args.max_subjects:
        nifti_files = nifti_files[:args.max_subjects]
    
    print(f"\nFound {len(nifti_files)} NIfTI files")
    
    # Load model
    net = TCM_AUXT(N=64)
    net = net.to(device)
    net.eval()
    
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('module.'):
                state_dict[k[7:]] = v
            else:
                state_dict[k] = v
        
        net.load_state_dict(state_dict)
        print("Model loaded successfully")
    
    if args.real:
        print("Updating entropy bottleneck...")
        net.update(force=True)
        print("Entropy model updated")
    
    # Process files
    subject_metrics = {}
    
    print("\nProcessing NIfTI files...")
    for file_idx, nifti_path in enumerate(nifti_files, 1):
        basename = os.path.basename(nifti_path)
        subject_id = basename.replace('.nii.gz', '').replace('.nii', '')
        subject_id = subject_id.replace('_cropped', '').replace('_roi', '')
        
        print(f"\n[{file_idx}/{len(nifti_files)}] {subject_id}")
        
        # Get bbox and cropping info
        bbox_info = bbox_dict.get(subject_id) if args.is_roi else None
        cropping_time = cropping_times.get(basename, 0.0)
        
        # Find original file path for ROI mode
        original_path = None
        if args.is_roi and args.original_dir:
            for ext in ['.nii.gz', '.nii']:
                candidate = os.path.join(args.original_dir, f"{subject_id}{ext}")
                if os.path.exists(candidate):
                    original_path = candidate
                    break
        
        try:
            result = process_nifti_file(
                nifti_path=nifti_path,
                net=net,
                device=device,
                args=args,
                bbox_info=bbox_info,
                original_nifti_path=original_path,
                cropping_time=cropping_time,
                target_slice_idx=args.target_slice
            )
            
            if result:
                subject_metrics[subject_id] = result
                
                print(f"  PSNR={result['avg_psnr']:.2f}dB, "
                      f"BPP={result['bpp']:.4f}, "
                      f"CR={result['compression_ratio']:.2f}x")
                
                # Save ROI metadata
                if result['roi_metadata']:
                    metadata_path = os.path.join(args.output, f'{subject_id}_roi_metadata.bin')
                    result['roi_metadata'].save(metadata_path)
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate averages
    if len(subject_metrics) == 0:
        print("\nNo subjects processed successfully")
        return
    
    avg_psnr = np.mean([m['avg_psnr'] for m in subject_metrics.values()])
    avg_ssim = np.mean([m['avg_ssim'] for m in subject_metrics.values()])
    avg_bpp = np.mean([m['bpp'] for m in subject_metrics.values()])
    avg_cr = np.mean([m['compression_ratio'] for m in subject_metrics.values()])
    avg_metadata_size = np.mean([m['metadata_size'] for m in subject_metrics.values()])
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Dataset: {args.data}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Subjects: {len(subject_metrics)}")
    print(f"ROI mode: {args.is_roi}")
    print(f"Axis: {args.axis} ({'sagittal' if args.axis==0 else 'coronal' if args.axis==1 else 'axial'})")
    print("-"*80)
    print(f"Average PSNR (dB):         {avg_psnr:.2f}")
    print(f"Average SSIM:              {avg_ssim:.4f}")
    print(f"Average BPP:               {avg_bpp:.4f}")
    print(f"Average Compression Ratio: {avg_cr:.2f}x")
    if args.is_roi:
        print(f"Average Metadata Size:     {avg_metadata_size:.0f} bytes")
    print("="*80)
    
    # Save JSON results
    result_file = os.path.join(args.output, 'evaluation_results.json')
    with open(result_file, 'w') as f:
        serializable_results = {}
        for k, v in subject_metrics.items():
            serializable_results[k] = {
                key: val for key, val in v.items() 
                if key not in ['affine', 'roi_metadata', 'compressed_slices']
            }
        
        json.dump({
            'dataset': args.data,
            'checkpoint': args.checkpoint,
            'num_subjects': len(subject_metrics),
            'is_roi': args.is_roi,
            'axis': args.axis,
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim,
            'avg_bpp': avg_bpp,
            'avg_compression_ratio': avg_cr,
            'avg_metadata_size': avg_metadata_size,
            'results': serializable_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {result_file}")
    
    # Save CSV results
    csv_file = os.path.join(args.output, 'subject_results.csv')
    with open(csv_file, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        
        # Header
        header = ['Subject_ID', 'Num_Slices', 'Avg_PSNR_dB', 'Avg_SSIM', 
                 'BPP', 'Compression_Ratio', 'Compressed_Size_Bytes']
        if args.is_roi:
            header.extend(['Metadata_Size_Bytes', 'Total_Size_Bytes', 'Cropping_Time_s'])
        header.extend(['Compression_Time_s', 'Decompression_Time_s'])
        writer.writerow(header)
        
        # Data
        for subject_id in sorted(subject_metrics.keys()):
            m = subject_metrics[subject_id]
            row = [
                subject_id,
                m['num_slices'],
                f"{m['avg_psnr']:.2f}",
                f"{m['avg_ssim']:.4f}",
                f"{m['bpp']:.4f}",
                f"{m['compression_ratio']:.2f}",
                f"{m['compressed_size']:.0f}",
            ]
            if args.is_roi:
                row.extend([
                    f"{m['metadata_size']}",
                    f"{m['total_compressed_size']:.0f}",
                    f"{m['cropping_time']:.4f}",
                ])
            row.extend([
                f"{m['compression_time']:.4f}",
                f"{m['decompression_time']:.4f}"
            ])
            writer.writerow(row)
        
        # Average row
        avg_row = ['AVERAGE', '-', f"{avg_psnr:.2f}", f"{avg_ssim:.4f}",
                   f"{avg_bpp:.4f}", f"{avg_cr:.2f}", '-']
        if args.is_roi:
            avg_row.extend([f"{avg_metadata_size:.0f}", '-', '-'])
        avg_row.extend(['-', '-'])
        writer.writerow(avg_row)
    
    print(f"CSV saved to: {csv_file}")


if __name__ == "__main__":
    main(sys.argv[1:])