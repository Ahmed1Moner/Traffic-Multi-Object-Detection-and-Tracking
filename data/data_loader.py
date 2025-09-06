import os
import glob
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from config.config import Config

class DataLoader:
    def __init__(self, config):
        self.config = config
        
    def prepare_dataset(self):
        """Prepare the dataset in YOLO format"""
        # Create directory structure
        (Path(self.config.OUTPUT_DIR) / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (Path(self.config.OUTPUT_DIR) / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (Path(self.config.OUTPUT_DIR) / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (Path(self.config.OUTPUT_DIR) / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        
        # Get all sequences
        all_sequences = self._get_sequences_with_labels()
        
        # Split sequences
        train_sequences, val_sequences = self._split_sequences(all_sequences)
        
        # Copy files
        self._copy_files(train_sequences, 'train')
        self._copy_files(val_sequences, 'val')
        
        # Create data.yaml
        self._create_data_yaml()
        
    def _get_sequences_with_labels(self):
        """Get sequences that have both images and labels"""
        sequences = set()
        original_img_dir = Path(self.config.IMAGE_DIR)
        
        for seq_dir in original_img_dir.iterdir():
            if seq_dir.is_dir():
                seq_name = seq_dir.name
                label_files = list(Path('/home/yazan/labels').rglob(f'{seq_name}*.txt'))
                if label_files:
                    sequences.add(seq_name)
        
        return sorted(sequences)
    
    def _split_sequences(self, sequences):
        """Split sequences into train and validation sets"""
        import random
        random.seed(42)
        random.shuffle(sequences)
        split_idx = int(0.8 * len(sequences))
        return sequences[:split_idx], sequences[split_idx:]
    
    def _copy_files(self, sequences, split):
        """Copy image and label files for the given split"""
        for seq in sequences:
            # Copy images
            seq_img_dir = Path(self.config.IMAGE_DIR) / seq
            for img_file in seq_img_dir.glob('*.jpg'):
                shutil.copy2(img_file, Path(self.config.OUTPUT_DIR) / 'images' / split / img_file.name)
            
            # Copy labels
            for label_file in Path('/home/yazan/labels').rglob(f'{seq}*.txt'):
                label_parts = label_file.stem.split('_')
                if len(label_parts) >= 3:
                    image_number = f"{label_parts[2]}.txt"
                    new_label_path = Path(self.config.OUTPUT_DIR) / 'labels' / split / image_number
                    shutil.copy2(label_file, new_label_path)
    
    def _create_data_yaml(self):
        """Create the data.yaml file for YOLO training"""
        data_yaml_content = f"""
path: {self.config.OUTPUT_DIR}
train: images/train
val: images/val
test: images/val

names:
  0: others
  1: car
  2: van
  3: bus
"""
        
        with open(Path(self.config.OUTPUT_DIR) / 'data.yaml', 'w') as f:
            f.write(data_yaml_content)