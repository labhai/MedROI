import os
import glob
import pickle
import numpy as np
import nibabel as nib
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse

def calculate_metrics(original_file, decompressed_file, compressed_file, metadata_file):
    """
    Calculate compression metrics for a single subject
    """
    results = {}
    
    # 1. Load metadata
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    results['subject_id'] = metadata['subject_id']
    results['compression_time'] = metadata['compression_time']
    
    # 2. Calculate file sizes
    original_size = os.path.getsize(original_file)
    compressed_size = os.path.getsize(compressed_file)
    
    results['original_size_MB'] = original_size / (1024 * 1024)
    results['compressed_size_MB'] = compressed_size / (1024 * 1024)
    
    # 3. Calculate Compression Ratio
    results['compression_ratio'] = original_size / compressed_size
    
    # 4. Load images for quality metrics
    original_nii = nib.load(original_file)
    original_img = original_nii.get_fdata()
    
    decompressed_nii = nib.load(decompressed_file)
    decompressed_img = decompressed_nii.get_fdata()
    
    # Handle 3D/4D data
    if len(original_img.shape) == 3:
        original_img = original_img[..., np.newaxis]
    if len(decompressed_img.shape) == 3:
        decompressed_img = decompressed_img[..., np.newaxis]
    
    sx, sy, sz, vols = original_img.shape
    
    # 5. Calculate BPP (Bits Per Pixel)
    # Total pixels = sx * sy * sz (for all slices)
    total_pixels = sx * sy * sz
    results['bpp'] = (compressed_size * 8) / total_pixels
    
    # 6. Calculate PSNR and SSIM
    # Calculate per-slice and then average
    psnr_values = []
    ssim_values = []
    
    for slice_idx in range(sz):
        for vol_idx in range(vols):
            orig_slice = original_img[:, :, slice_idx, vol_idx]
            dec_slice = decompressed_img[:, :, slice_idx, vol_idx]
            
            # PSNR
            data_range = orig_slice.max() - orig_slice.min()
            if data_range > 0:
                slice_psnr = psnr(orig_slice, dec_slice, data_range=data_range)
                psnr_values.append(slice_psnr)
                
                # SSIM
                slice_ssim = ssim(orig_slice, dec_slice, data_range=data_range)
                ssim_values.append(slice_ssim)
    
    results['psnr_db'] = np.mean(psnr_values)
    results['psnr_std'] = np.std(psnr_values)
    results['ssim'] = np.mean(ssim_values)
    results['ssim_std'] = np.std(ssim_values)
    
    # Additional info
    results['image_shape'] = f"{sx}x{sy}x{sz}x{vols}"
    results['num_slices'] = sz
    results['num_volumes'] = vols
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate SirenMRI compression results')
    parser.add_argument('-o', '--output_dir', required=True, 
                        help='Directory containing compressed subjects (e.g., ./2D_output)')
    parser.add_argument('-i', '--input_dir', required=True,
                        help='Directory containing original NIfTI files (e.g., /mnt/nas-data/ADNI200/nifti)')
    parser.add_argument('-r', '--results_file', default='compression_results.csv',
                        help='Output CSV file name')
    
    args = parser.parse_args()
    
    # Find all subject directories
    subject_dirs = sorted([d for d in glob.glob(os.path.join(args.output_dir, '*')) 
                          if os.path.isdir(d) and not d.endswith('__pycache__')])
    
    print(f"Found {len(subject_dirs)} subjects to evaluate")
    print("="*80)
    
    all_results = []
    
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        
        # Define file paths
        compressed_file = os.path.join(subject_dir, 'SirenCompression.7z')
        decompressed_file = os.path.join(subject_dir, 'dwi_decompressed.nii.gz')
        metadata_file = os.path.join(subject_dir, 'metadata.pickle')
        
        # Find original file
        original_file_gz = os.path.join(args.input_dir, f'{subject_id}.nii.gz')
        original_file_nii = os.path.join(args.input_dir, f'{subject_id}.nii')
        
        if os.path.exists(original_file_gz):
            original_file = original_file_gz
        elif os.path.exists(original_file_nii):
            original_file = original_file_nii
        else:
            print(f"⚠ Warning: Original file not found for {subject_id}, skipping...")
            continue
        
        # Check if all required files exist
        if not os.path.exists(compressed_file):
            print(f"⚠ Warning: Compressed file not found for {subject_id}, skipping...")
            continue
        if not os.path.exists(decompressed_file):
            print(f"⚠ Warning: Decompressed file not found for {subject_id}, skipping...")
            continue
        if not os.path.exists(metadata_file):
            print(f"⚠ Warning: Metadata file not found for {subject_id}, skipping...")
            continue
        
        print(f"Evaluating: {subject_id}")
        
        try:
            results = calculate_metrics(original_file, decompressed_file, 
                                       compressed_file, metadata_file)
            all_results.append(results)
            
            print(f"  ✓ PSNR: {results['psnr_db']:.2f} dB")
            print(f"  ✓ SSIM: {results['ssim']:.4f}")
            print(f"  ✓ Compression Ratio: {results['compression_ratio']:.2f}x")
            print(f"  ✓ BPP: {results['bpp']:.4f}")
            print(f"  ✓ Time: {results['compression_time']:.2f} s")
            print()
            
        except Exception as e:
            print(f"✗ Error processing {subject_id}: {str(e)}")
            continue
    
    # Convert to DataFrame and save
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Reorder columns for better readability
        column_order = [
            'subject_id', 'image_shape', 'num_slices', 'num_volumes',
            'psnr_db', 'psnr_std', 'ssim', 'ssim_std',
            'compression_ratio', 'bpp', 
            'original_size_MB', 'compressed_size_MB',
            'compression_time'
        ]
        df = df[column_order]
        
        # Save to CSV
        output_path = os.path.join(args.output_dir, args.results_file)
        df.to_csv(output_path, index=False)
        
        print("="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"\nNumber of subjects: {len(df)}")
        print(f"\nAverage PSNR: {df['psnr_db'].mean():.2f} ± {df['psnr_db'].std():.2f} dB")
        print(f"Average SSIM: {df['ssim'].mean():.4f} ± {df['ssim'].std():.4f}")
        print(f"Average Compression Ratio: {df['compression_ratio'].mean():.2f}x ± {df['compression_ratio'].std():.2f}x")
        print(f"Average BPP: {df['bpp'].mean():.4f} ± {df['bpp'].std():.4f}")
        print(f"Average Time: {df['compression_time'].mean():.2f} ± {df['compression_time'].std():.2f} s")
        print(f"\nResults saved to: {output_path}")
        print("="*80)
        
    else:
        print("No results to save!")

if __name__ == '__main__':
    main()

"""
Usage example:
python evaluate.py \
    -o ./2D_output \
    -i /mnt/nas-data/ADNI200/nifti \
    -r compression_results.csv
"""