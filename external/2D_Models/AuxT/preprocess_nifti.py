# preprocess_medical_data.py
import nibabel as nib
import numpy as np
from PIL import Image
import os
import json
from pathlib import Path
import argparse
from tqdm import tqdm

def nifti_to_png_slices_per_subject(nifti_path, output_dir, subject_id=None, 
                                      axis='axial', normalize_method='percentile', 
                                      min_std=5.0):
    """
    한 피험자의 NIfTI 파일을 PNG 슬라이스로 변환
    
    Args:
        nifti_path: .nii 또는 .nii.gz 파일 경로
        output_dir: PNG 저장 디렉토리
        subject_id: 피험자 ID (파일명에 포함됨)
        min_std: 빈 슬라이스 필터링 기준 (표준편차)
    
    Returns:
        저장된 슬라이스 수
    """
    # NIfTI 로드
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    # subject_id 추출 (제공되지 않으면 파일명에서)
    if subject_id is None:
        subject_id = Path(nifti_path).stem.replace('.nii', '')
    
    print(f"처리 중: {subject_id}")
    print(f"  Shape: {data.shape}, 범위: [{data.min():.2f}, {data.max():.2f}]")
    
    # 정규화
    if normalize_method == 'minmax':
        data_min, data_max = data.min(), data.max()
        if data_max - data_min > 0:
            data_normalized = ((data - data_min) / (data_max - data_min) * 255)
        else:
            data_normalized = np.zeros_like(data)
    
    elif normalize_method == 'percentile':
        p_low, p_high = np.percentile(data, [1, 99])
        if p_high - p_low > 0:
            data_clipped = np.clip(data, p_low, p_high)
            data_normalized = ((data_clipped - p_low) / (p_high - p_low) * 255)
        else:
            data_normalized = np.zeros_like(data)
    
    elif normalize_method == 'window':
        window_center = np.median(data)
        window_width = np.std(data) * 4
        if window_width > 0:
            data_normalized = np.clip(
                ((data - (window_center - window_width/2)) / window_width) * 255,
                0, 255
            )
        else:
            data_normalized = np.zeros_like(data)
    
    data_normalized = data_normalized.astype(np.uint8)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 슬라이스 저장
    saved_count = 0
    
    if axis == 'axial':  # Z축 (가장 일반적)
        num_slices = data_normalized.shape[2]
        for i in range(num_slices):
            slice_data = data_normalized[:, :, i]
            
            # 빈 슬라이스 필터링 (옵션)
            if slice_data.std() > min_std:
                slice_img = Image.fromarray(slice_data)
                # 파일명에 피험자 ID 포함
                slice_img.save(f"{output_dir}/{subject_id}_slice_{i:04d}.png")
                saved_count += 1
    
    elif axis == 'sagittal':  # X축
        num_slices = data_normalized.shape[0]
        for i in range(num_slices):
            slice_data = data_normalized[i, :, :]
            if slice_data.std() > min_std:
                slice_img = Image.fromarray(slice_data)
                slice_img.save(f"{output_dir}/{subject_id}_slice_{i:04d}.png")
                saved_count += 1
    
    elif axis == 'coronal':  # Y축
        num_slices = data_normalized.shape[1]
        for i in range(num_slices):
            slice_data = data_normalized[:, i, :]
            if slice_data.std() > min_std:
                slice_img = Image.fromarray(slice_data)
                slice_img.save(f"{output_dir}/{subject_id}_slice_{i:04d}.png")
                saved_count += 1
    
    print(f"  ✅ {saved_count}/{num_slices} 슬라이스 저장 (빈 슬라이스 제외)")
    
    return saved_count


def split_subjects_train_test(nifti_files, train_ratio=0.8, seed=42):
    """
    피험자를 train/test로 분리
    
    Args:
        nifti_files: NIfTI 파일 리스트
        train_ratio: train 데이터 비율
        seed: random seed
    
    Returns:
        train_files, test_files
    """
    import random
    random.seed(seed)
    
    files = list(nifti_files)
    random.shuffle(files)
    
    split_idx = int(len(files) * train_ratio)
    train_files = files[:split_idx]
    test_files = files[split_idx:]
    
    return train_files, test_files


def process_dataset(input_dir, output_dir, split_name, nifti_files,
                    axis='axial', normalize_method='percentile', min_std=5.0):
    """
    여러 NIfTI 파일을 한 번에 처리
    
    Args:
        input_dir: 입력 디렉토리
        output_dir: 출력 디렉토리
        split_name: 'train' 또는 'kodak'
        nifti_files: 처리할 NIfTI 파일 리스트
    """
    output_path = Path(output_dir) / split_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"{split_name.upper()} 데이터셋 처리: {len(nifti_files)}개 피험자")
    print(f"{'='*70}\n")
    
    total_slices = 0
    total_subjects = 0
    
    for nifti_file in tqdm(nifti_files, desc=f"Processing {split_name}"):
        try:
            num_slices = nifti_to_png_slices_per_subject(
                str(nifti_file),
                str(output_path),
                subject_id=nifti_file.stem.replace('.nii', ''),
                axis=axis,
                normalize_method=normalize_method,
                min_std=min_std
            )
            total_slices += num_slices
            total_subjects += 1
        except Exception as e:
            print(f"  ❌ 오류 발생: {nifti_file.name} - {e}")
            continue
    
    # 메타데이터 저장
    metadata = {
        'total_subjects': total_subjects,
        'total_slices': total_slices,
        'axis': axis,
        'normalize_method': normalize_method,
        'min_std': min_std,
        'subjects': [f.stem.replace('.nii', '') for f in nifti_files]
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"{split_name.upper()} 완료:")
    print(f"  피험자 수: {total_subjects}")
    print(f"  총 슬라이스: {total_slices}")
    print(f"  평균 슬라이스/피험자: {total_slices/max(total_subjects, 1):.1f}")
    print(f"  저장 위치: {output_path}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='의료 영상 NIfTI to PNG 전처리')
    parser.add_argument('-i', '--input_dir', required=True, 
                        help='NIfTI 파일이 있는 디렉토리')
    parser.add_argument('-o', '--output_dir', required=True, 
                        help='PNG 저장 디렉토리')
    parser.add_argument('--axis', default='axial', 
                        choices=['axial', 'sagittal', 'coronal'])
    parser.add_argument('--normalize', default='percentile', 
                        choices=['minmax', 'percentile', 'window'])
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Train 데이터 비율 (default: 0.8)')
    parser.add_argument('--min-std', type=float, default=5.0,
                        help='빈 슬라이스 필터링 기준 (default: 5.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--ext', default='.nii*',
                        help='파일 확장자 패턴 (default: .nii*)')
    
    args = parser.parse_args()
    
    # NIfTI 파일 찾기
    input_path = Path(args.input_dir)
    nifti_files = list(input_path.glob(f"*{args.ext}"))[:100]
    
    if len(nifti_files) == 0:
        print(f"❌ {args.input_dir}에서 {args.ext} 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n총 {len(nifti_files)}개의 NIfTI 파일 발견")
    
    # Train/Test 분리
    train_files, test_files = split_subjects_train_test(
        nifti_files, 
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    print(f"Train: {len(train_files)}명, Test: {len(test_files)}명")
    
    # Train 데이터 처리
    process_dataset(
        args.input_dir,
        args.output_dir,
        'train',
        train_files,
        axis=args.axis,
        normalize_method=args.normalize,
        min_std=args.min_std
    )
    
    # Test 데이터 처리 (kodak 이름 사용 - train.py와 호환)
    process_dataset(
        args.input_dir,
        args.output_dir,
        'kodak',  # train.py에서 'kodak' split 사용
        test_files,
        axis=args.axis,
        normalize_method=args.normalize,
        min_std=args.min_std
    )
    
    print("\n" + "="*70)
    print("✅ 전체 전처리 완료!")
    print("="*70)
    print(f"\n다음 명령으로 학습 시작:")
    print(f"python train.py -d {args.output_dir} --lmbda 0.0483 --cuda --save\n")


if __name__ == "__main__":
    main()
    
'''
python preprocess_nifti.py \
  -i /home/jiwon/MRI-COMPRESSION/bbox_outputs/cropped_volumes \
  -o /home/jiwon/MRI-COMPRESSION/AuxT/preprocessed_data/cropped \
  --train-ratio 1.0
'''