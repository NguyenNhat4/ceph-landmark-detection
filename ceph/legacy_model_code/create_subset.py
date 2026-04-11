import os
import shutil
from pathlib import Path

def create_subset(data_dir='./data', n_train=50, n_val=10):
    """
    Tạo subset nhỏ từ dataset gốc
    """
    # Tạo thư mục subset
    subset_dir = Path(data_dir) / 'subset_50'
    subset_dir.mkdir(exist_ok=True)

    for mode in ['train', 'valid']:
        n_samples = n_train if mode == 'train' else n_val

        # Paths
        src_img_dir = Path(data_dir) / mode / 'Cephalograms'
        src_anno_dir = Path(data_dir) / mode / 'Annotations' / 'Cephalometric Landmarks' / 'Senior Orthodontists'

        dst_img_dir = subset_dir / mode / 'Cephalograms'
        dst_anno_dir = subset_dir / mode / 'Annotations' / 'Cephalometric Landmarks' / 'Senior Orthodontists'

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_anno_dir.mkdir(parents=True, exist_ok=True)

        # Copy n_samples files
        img_files = sorted(list(src_img_dir.glob('*.png')))[:n_samples]

        for img_file in img_files:
            # Copy image
            shutil.copy(img_file, dst_img_dir / img_file.name)

            # Copy annotation
            anno_file = src_anno_dir / img_file.with_suffix('.json').name
            if anno_file.exists():
                shutil.copy(anno_file, dst_anno_dir / anno_file.name)

        print(f"Created {mode}: {len(img_files)} images")

    # Copy CSV
    shutil.copy(
        Path(data_dir) / 'cephalogram_machine_mappings.csv',
        subset_dir / 'cephalogram_machine_mappings.csv'
    )

    print(f"\nSubset created at: {subset_dir}")

if __name__ == '__main__':
    create_subset(n_train=50, n_val=10)