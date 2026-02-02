"""
SirenMRI 3D Medical Image Compression Evaluation with ROI Support

This script evaluates SirenMRI (Implicit Neural Representation) compression
on 3D medical imaging datasets with support for ROI-based compression.

Features:
- SirenMRI (Implicit Neural Representation) compression
- ROI (Region of Interest) compression with metadata
- 3D volume processing with automatic memory management
- Quality metrics: PSNR, SSIM
- Compression metrics: BPP, Compression Ratio
- Batch processing support

Reference: SIREN - Implicit Neural Representations with Periodic Activation Functions
Author: [Your Name]
License: MIT
"""

import scipy.io
import argparse
import os
import random
import sys
import torch
import util
import numpy as np
import nibabel as nib
from siren import Siren
from training import Trainer
from sklearn.preprocessing import MinMaxScaler
import pickle
import time
import glob
import json
import pandas as pd
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import gc

# Argument parser
parser = argparse.ArgumentParser(
    description='SirenMRI 3D Medical Image Compression',
    formatter_class=argparse.RawDescriptionHelpFormatter
)

# Model parameters
parser.add_argument('--layer_size', type=int, default=256,
                   help='Hidden layer size (default: 256)')
parser.add_argument('--num_layers', type=int, default=3,
                   help='Number of layers (default: 3)')
parser.add_argument('--w0', type=float, default=30.0,
                   help='w0 parameter for SIREN (default: 30.0)')
parser.add_argument('--w0_initial', type=float, default=30.0,
                   help='w0 for first layer (default: 30.0)')

# Training parameters
parser.add_argument('--num_iters', type=int, default=2000,
                   help='Training iterations (default: 2000)')
parser.add_argument('--learning_rate', type=float, default=2e-4,
                   help='Learning rate (default: 2e-4)')
parser.add_argument('--seed', type=int, default=random.randint(1, int(1e6)),
                   help='Random seed')

# Data paths
parser.add_argument('--image', help='Single NIfTI file to compress')
parser.add_argument('--data_dir', default='',
                   help='Directory with multiple NIfTI files')
parser.add_argument('--original_dir', default='',
                   help='Directory with original full files (for ROI mode)')
parser.add_argument('--original_full', default='',
                   help='Original full file (for single file mode)')

# Output
parser.add_argument('--logdir', default='./siren_results',
                   help='Output directory (default: ./siren_results)')
parser.add_argument('--log_measures', action='store_true',
                   help='Save measures for each epoch')

# Optional
parser.add_argument('--bbox_csv', help='CSV with 3D bbox info')
parser.add_argument('--max_subjects', type=int, default=None,
                   help='Maximum subjects to process')

args = parser.parse_args()

# Setup torch and CUDA
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


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
        
        # Bounding Box
        if self.bbox:
            bbox_array = np.array([
                self.bbox['x_min'], self.bbox['x_max'],
                self.bbox['y_min'], self.bbox['y_max'],
                self.bbox['z_min'], self.bbox['z_max']
            ], dtype=np.int16)
            data += bbox_array.tobytes()
        else:
            data += np.zeros(6, dtype=np.int16).tobytes()
        
        # Original Shape
        if self.original_shape:
            shape_array = np.array(self.original_shape, dtype=np.int16)
            data += shape_array.tobytes()
        else:
            data += np.zeros(3, dtype=np.int16).tobytes()
        
        # Affine Matrix (3×3 only)
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


# ========== Quality Evaluation ==========

class CompressionEvaluator:
    """Compression quality evaluator with ROI support"""
    
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
            max_val = max(original.max(), reconstructed.max())
            if max_val == 0:
                max_val = 1.0
            psnr = 20 * np.log10(max_val / np.sqrt(mse))
        
        # SSIM (slice-wise average)
        ssim_values = []
        
        if len(original.shape) == 3:
            for i in range(original.shape[2]):
                slice_orig = original[:, :, i]
                slice_recon = reconstructed[:, :, i]
                if slice_orig.std() > 0 and slice_recon.std() > 0:
                    data_range = max(slice_orig.max(), slice_recon.max()) - min(slice_orig.min(), slice_recon.min())
                    if data_range > 0:
                        ssim_val = ssim(slice_orig, slice_recon, data_range=data_range)
                        ssim_values.append(ssim_val)
        elif len(original.shape) == 4:
            for v in range(original.shape[3]):
                for i in range(original.shape[2]):
                    slice_orig = original[:, :, i, v]
                    slice_recon = reconstructed[:, :, i, v]
                    if slice_orig.std() > 0 and slice_recon.std() > 0:
                        data_range = max(slice_orig.max(), slice_recon.max()) - min(slice_orig.min(), slice_recon.min())
                        if data_range > 0:
                            ssim_val = ssim(slice_orig, slice_recon, data_range=data_range)
                            ssim_values.append(ssim_val)
        
        avg_ssim = np.mean(ssim_values) if ssim_values else 0.0
        
        return mse, psnr, avg_ssim
    
    @staticmethod
    def evaluate_volume(original_cropped_volume, reconstructed, model_size_bytes,
                       comp_time, original_full_volume=None, cropping_time=0,
                       original_full_file_path=None, bbox_info=None, roi_metadata=None):
        """
        Evaluate 3D volume compression.
        
        Args:
            original_cropped_volume: cropped ROI volume
            reconstructed: decompressed volume
            model_size_bytes: model size in bytes
            comp_time: compression time
            original_full_volume: original full volume
            cropping_time: cropping time
            original_full_file_path: original file path
            bbox_info: bbox coordinates
            roi_metadata: ROIMetadata object
        """
        
        # Apply full restoration if ROI
        if original_full_volume is not None and bbox_info is not None:
            print("\nApplying full restoration scenario...")
            
            restored_full = np.zeros_like(original_full_volume, dtype=reconstructed.dtype)
            
            x_min, x_max = bbox_info['x_min'], bbox_info['x_max']
            y_min, y_max = bbox_info['y_min'], bbox_info['y_max']
            z_min, z_max = bbox_info['z_min'], bbox_info['z_max']
            
            expected_shape = (x_max - x_min, y_max - y_min, z_max - z_min)
            actual_shape = reconstructed.shape
            
            if expected_shape != actual_shape:
                print(f"Shape mismatch: expected {expected_shape}, got {actual_shape}")
                x_max = x_min + actual_shape[0]
                y_max = y_min + actual_shape[1]
                z_max = z_min + actual_shape[2]
            
            restored_full[x_min:x_max, y_min:y_max, z_min:z_max] = reconstructed
            
            mse, psnr, ssim_val = CompressionEvaluator.calculate_metrics_3d(
                original_full_volume, restored_full
            )
            
            print(f"Evaluated with full restoration")
        else:
            mse, psnr, ssim_val = CompressionEvaluator.calculate_metrics_3d(
                original_cropped_volume, reconstructed
            )
        
        # Metadata size
        metadata_size = ROIMetadata.TOTAL_SIZE if roi_metadata is not None else 0
        
        # Compression ratio (memory-based)
        if original_full_volume is not None:
            original_memory_size = original_full_volume.size * original_full_volume.itemsize
        else:
            original_memory_size = original_cropped_volume.size * original_cropped_volume.itemsize
        
        total_compressed_size = model_size_bytes + metadata_size
        compression_ratio_memory = original_memory_size / total_compressed_size if total_compressed_size > 0 else 0
        
        # Compression ratio (file-based)
        compression_ratio_file = None
        original_file_size = None
        
        if original_full_file_path and os.path.exists(original_full_file_path):
            original_file_size = os.path.getsize(original_full_file_path)
            compression_ratio_file = original_file_size / total_compressed_size
        
        # BPP
        total_voxels = original_full_volume.size if original_full_volume is not None else original_cropped_volume.size
        bpp = (total_compressed_size * 8) / total_voxels
        
        # Total time
        decomp_time = 0.1
        total_processing_time = cropping_time + comp_time + decomp_time
        
        return {
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim_val,
            'compression_ratio': compression_ratio_memory,
            'compression_ratio_file_based': compression_ratio_file,
            'bpp': bpp,
            'original_full_size': original_memory_size,
            'original_file_size': original_file_size,
            'cropped_size': original_cropped_volume.size * original_cropped_volume.itemsize,
            'model_size': model_size_bytes,
            'metadata_size': metadata_size,
            'total_compressed_size': total_compressed_size,
            'cropping_time': cropping_time,
            'compression_time': comp_time,
            'decompression_time': decomp_time,
            'total_time': total_processing_time
        }


# ========== Utility Functions ==========

def load_bbox_info(bbox_csv_path):
    """Load 3D bounding box information from CSV"""
    if not os.path.exists(bbox_csv_path):
        print(f"BBox file not found: {bbox_csv_path}")
        return {}
    
    try:
        df = pd.read_csv(bbox_csv_path)
        bbox_dict = {}
        
        print(f"Loading BBox info from {bbox_csv_path}")
        
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
        
        print(f"Loaded BBox for {len(bbox_dict)} subjects")
        return bbox_dict
        
    except Exception as e:
        print(f"Failed to load BBox: {e}")
        return {}


def find_original_file(cropped_filename, original_dir):
    """Find original file from cropped filename"""
    if not original_dir or not os.path.exists(original_dir):
        return None
    
    basename = os.path.basename(cropped_filename)
    
    if basename.endswith('.nii.gz'):
        subject_id = basename[:-7]
    elif basename.endswith('.nii'):
        subject_id = basename[:-4]
    else:
        subject_id = basename
    
    subject_id = subject_id.replace('_cropped', '').replace('_roi', '')
    
    possible_extensions = ['.nii', '.nii.gz']
    for ext in possible_extensions:
        candidate = os.path.join(original_dir, f"{subject_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    
    return None


def cleanup_memory():
    """Clean up GPU and CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# ========== Main Processing ==========

def main():
    # Load bbox info
    bbox_dict = {}
    if args.bbox_csv:
        bbox_dict = load_bbox_info(args.bbox_csv)
    
    # Results storage
    results = {'fp_bpp': [], 'fp_psnr': []}
    all_results = []
    
    # Get list of NIfTI files
    if args.data_dir:
        print(f"Processing directory: {args.data_dir}")
        
        nifti_files = (glob.glob(os.path.join(args.data_dir, "*.nii.gz")) + 
                      glob.glob(os.path.join(args.data_dir, "*.nii")))
        nifti_files = sorted(nifti_files)
        
        if len(nifti_files) == 0:
            print(f"No NIfTI files found in {args.data_dir}")
            sys.exit(1)
        
        if args.max_subjects:
            nifti_files = nifti_files[:args.max_subjects]
        
        print(f"Found {len(nifti_files)} files")
    else:
        nifti_files = [args.image]
    
    # Process each file
    for file_idx, img_file in enumerate(nifti_files):
        subject_start_time = time.time()
        
        subject_id = os.path.basename(img_file).replace('.nii.gz', '').replace('.nii', '')
        subject_logdir = os.path.join(args.logdir, subject_id)
        
        print(f"\n{'='*80}")
        print(f"Processing [{file_idx+1}/{len(nifti_files)}]: {subject_id}")
        print(f"{'='*80}")
        
        if not os.path.exists(subject_logdir):
            os.makedirs(subject_logdir)
        
        # Load ROI image
        print(f"Loading: {img_file}")
        img_nib = nib.load(img_file)
        new_header = img_nib.header.copy()
        img_tmp = img_nib.get_fdata()
        
        if len(img_tmp.shape) == 3:
            img_tmp = img_tmp[..., np.newaxis]
        
        print(f"ROI shape: {img_tmp.shape}")
        
        # Load original full volume
        original_full_volume = None
        original_full_path = None
        affine_matrix = None
        
        if args.data_dir and args.original_dir:
            original_full_path = find_original_file(img_file, args.original_dir)
        elif args.original_full:
            original_full_path = args.original_full
        elif args.data_dir and not args.original_dir:
            original_full_path = img_file
        
        if original_full_path and os.path.exists(original_full_path):
            print(f"Loading original: {original_full_path}")
            original_nii = nib.load(original_full_path)
            original_full_volume = original_nii.get_fdata()
            affine_matrix = original_nii.affine
            
            if len(original_full_volume.shape) == 3:
                original_full_volume = original_full_volume[..., np.newaxis]
            
            print(f"Original shape: {original_full_volume.shape}")
        
        # Get bbox info
        bbox_info = bbox_dict.get(subject_id)
        if bbox_info:
            print(f"Found bbox for {subject_id}")
        
        # Create ROI metadata
        roi_metadata = None
        if bbox_info and original_full_volume is not None and affine_matrix is not None:
            roi_metadata = ROIMetadata(
                bbox=bbox_info,
                original_shape=original_full_volume.shape[:3],
                affine_matrix=affine_matrix
            )
            print("ROI Metadata created")
        
        # Prepare data
        img = img_tmp[:,:,:,:]
        img = np.transpose(img, (3, 0, 1, 2))
        img = torch.from_numpy(img.astype(np.float32))
        
        num_vols = img_tmp.shape[3]
        
        # Setup model
        func_rep = Siren(
            dim_in=3,
            dim_hidden=args.layer_size,
            dim_out=num_vols,
            num_layers=args.num_layers,
            final_activation=torch.nn.Identity(),
            w0_initial=args.w0_initial,
            w0=args.w0
        )
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            func_rep = torch.nn.DataParallel(func_rep)
        func_rep.to(device)
        
        # Setup training
        trainer = Trainer(func_rep, lr=args.learning_rate)
        coordinates, features = util.to_coordinates_and_features_3D(img)
        
        scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
        scaler.fit(features.numpy())
        features = scaler.transform(features.numpy())
        features = torch.from_numpy(features.astype(np.float32))
        
        coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)
        
        # Calculate model size
        model_size_kb = util.model_size_in_bits(func_rep) / 8000.
        model_size_bytes = util.model_size_in_bits(func_rep) / 8.
        print(f'Model size: {model_size_kb:.1f}kB')
        
        fp_bpp = util.bpp(model=func_rep, image=img)
        print(f'BPP: {fp_bpp:.2f}')
        
        # Train model
        print("\nTraining...")
        trainer.train(coordinates, features, num_iters=args.num_iters)
        print(f'Best PSNR: {trainer.best_vals["psnr"]:.2f}')
        
        if args.log_measures:
            with open(subject_logdir + '/log.pickle', 'wb') as handle:
                pickle.dump(trainer.logs, handle)
        
        results['fp_bpp'].append(fp_bpp)
        results['fp_psnr'].append(trainer.best_vals['psnr'])
        
        # Save model
        torch.save(trainer.best_model, subject_logdir + '/best_model.pt')
        
        indata = {'input': coordinates.cpu().numpy()}
        scipy.io.savemat(subject_logdir + '/input_to_best_model.mat', indata)
        
        # Save ROI metadata
        if roi_metadata is not None:
            metadata_path = subject_logdir + '/roi_metadata.bin'
            roi_metadata.save(metadata_path)
            print(f"ROI Metadata saved: {ROIMetadata.TOTAL_SIZE} bytes")
        
        # Generate decompressed image
        func_rep.load_state_dict(trainer.best_model)
        
        print("\nDecompressing...")
        with torch.no_grad():
            PredParams = func_rep(coordinates)
        
        PredParams = scaler.inverse_transform(PredParams.cpu().numpy())
        img_decompressed = np.reshape(PredParams, (img_tmp.shape[0], img_tmp.shape[1], 
                                                    img_tmp.shape[2], img_tmp.shape[3]))
        
        subject_end_time = time.time()
        compression_time = subject_end_time - subject_start_time
        
        # Evaluate
        print("\nEvaluating...")
        
        if num_vols == 1:
            original_eval = np.squeeze(img_tmp, axis=-1)
            decompressed_eval = np.squeeze(img_decompressed, axis=-1)
            if original_full_volume is not None and len(original_full_volume.shape) == 4:
                original_full_eval = np.squeeze(original_full_volume, axis=-1)
            else:
                original_full_eval = original_full_volume
        else:
            original_eval = img_tmp
            decompressed_eval = img_decompressed
            original_full_eval = original_full_volume
        
        metrics = CompressionEvaluator.evaluate_volume(
            original_cropped_volume=original_eval,
            reconstructed=decompressed_eval,
            model_size_bytes=model_size_bytes,
            comp_time=compression_time,
            original_full_volume=original_full_eval,
            cropping_time=0,
            original_full_file_path=original_full_path,
            bbox_info=bbox_info,
            roi_metadata=roi_metadata
        )
        
        # Save decompressed NIfTI
        if num_vols == 1:
            img_decompressed_save = np.squeeze(img_decompressed, axis=-1)
        else:
            img_decompressed_save = img_decompressed
        
        new_img = nib.nifti1.Nifti1Image(img_decompressed_save, None, header=new_header)
        compressed_file_path = subject_logdir + '/dwi_decompressed.nii.gz'
        nib.save(new_img, compressed_file_path)
        
        # Save results
        result_data = {
            'subject_id': subject_id,
            'filename': os.path.basename(img_file),
            'original_full_path': original_full_path if original_full_path else 'N/A',
            'cropped_shape': list(img_tmp.shape),
            'original_full_shape': list(original_full_volume.shape) if original_full_volume is not None else 'N/A',
            'compression_method': 'SirenMRI_3D',
            'has_roi_metadata': roi_metadata is not None,
            'metadata_size_bytes': ROIMetadata.TOTAL_SIZE if roi_metadata else 0,
            'model_params': {
                'layer_size': args.layer_size,
                'num_layers': args.num_layers,
                'w0': args.w0,
                'w0_initial': args.w0_initial,
                'num_iters': args.num_iters,
                'learning_rate': args.learning_rate
            },
            'seed': args.seed,
            'metrics': metrics
        }
        
        result_file = os.path.join(subject_logdir, f'{subject_id}_results.json')
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        csv_row = {
            'Subject_ID': subject_id,
            'PSNR_dB': f"{metrics['psnr']:.2f}",
            'SSIM': f"{metrics['ssim']:.4f}",
            'Compression_Ratio': f"{metrics['compression_ratio']:.2f}",
            'BPP': f"{metrics['bpp']:.4f}",
            'Metadata_Size_B': metrics['metadata_size'],
            'Model_Size_KB': f"{metrics['model_size']/1024:.2f}",
        }
        all_results.append(csv_row)
        
        print(f"\nCompleted: {subject_id}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  Compression Ratio: {metrics['compression_ratio']:.2f}:1")
        print(f"  BPP: {metrics['bpp']:.4f}")
        print(f"  Time: {compression_time:.2f}s")
        
        # Memory cleanup
        cleanup_memory()
        
        print(f"{'='*80}\n")
    
    # Save batch summary
    if len(nifti_files) > 1:
        print(f"\nCompleted {len(nifti_files)} subjects")
        
        if all_results:
            df = pd.DataFrame(all_results)
            summary_csv = os.path.join(args.logdir, 'batch_summary.csv')
            df.to_csv(summary_csv, index=False)
            print(f"\nSummary saved to: {summary_csv}")
            print(f"\n{df.to_string(index=False)}")
        
        print(f"\nAverage PSNR: {np.mean(results['fp_psnr']):.2f} dB")
        print(f"Average BPP: {np.mean(results['fp_bpp']):.2f}")


if __name__ == "__main__":
    main()