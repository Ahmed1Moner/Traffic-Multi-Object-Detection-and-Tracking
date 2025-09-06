import re
import cv2
import imageio
from pathlib import Path
from tqdm import tqdm
import numpy as np
from ultralytics import YOLO

def extract_frame_number(path):
    """
    Extract frame number from filename using regex
    
    Args:
        path (Path): Path object to extract frame number from
        
    Returns:
        int: Extracted frame number or 0 if not found
    """
    match = re.search(r'(\d+)', path.stem)
    return int(match.group(1)) if match else 0

def create_segment_gif(model, image_dir, output_path, start_frame=0, end_frame=100, conf=0.25):
    """
    Create GIF from a specific segment of a sequence
    
    Args:
        model: YOLO model for inference
        image_dir (str): Directory containing images
        output_path (str): Output GIF path
        start_frame (int): Starting frame index
        end_frame (int): Ending frame index
        conf (float): Confidence threshold
    """
    image_paths = list(Path(image_dir).glob('*.jpg'))
    image_paths.sort(key=extract_frame_number)
    image_paths = [str(p) for p in image_paths]
    
    segment_paths = image_paths[start_frame:end_frame]
    frames = []
    
    for img_path in tqdm(segment_paths, desc=f"Processing frames {start_frame}-{end_frame}"):
        results = model(img_path, conf=conf)
        annotated_frame = results[0].plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frames.append(annotated_frame_rgb)
    
    imageio.mimsave(output_path, frames, fps=10)
    return output_path

def create_threshold_gif(model, image_dir, output_path, confidence=0.25, max_frames=100):
    """
    Create GIF with specific confidence threshold
    
    Args:
        model: YOLO model for inference
        image_dir (str): Directory containing images
        output_path (str): Output GIF path
        confidence (float): Confidence threshold
        max_frames (int): Maximum number of frames to process
    """
    image_paths = list(Path(image_dir).glob('*.jpg'))
    image_paths.sort(key=extract_frame_number)
    image_paths = [str(p) for p in image_paths]
    
    frames = []
    for img_path in tqdm(image_paths[:max_frames], desc=f"Processing with conf={confidence}"):
        results = model(img_path, conf=confidence)
        annotated_frame = results[0].plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frames.append(annotated_frame_rgb)
    
    imageio.mimsave(output_path, frames, fps=10)
    return output_path

def create_processed_gif(model, image_dir, output_path, style="normal", max_frames=100, conf=0.25):
    """
    Create GIF with different processing styles
    
    Args:
        model: YOLO model for inference
        image_dir (str): Directory containing images
        output_path (str): Output GIF path
        style (str): Processing style ("normal", "birds_eye", "close_up")
        max_frames (int): Maximum number of frames to process
        conf (float): Confidence threshold
    """
    image_paths = list(Path(image_dir).glob('*.jpg'))
    image_paths.sort(key=extract_frame_number)
    image_paths = [str(p) for p in image_paths]
    
    frames = []
    for img_path in tqdm(image_paths[:max_frames], desc=f"Processing {style} view"):
        img = cv2.imread(img_path)
        
        if style == "normal":
            processed_img = img
        elif style == "birds_eye":
            height, width = img.shape[:2]
            src = np.float32([[width*0.3, height*0.65], [width*0.7, height*0.65],
                             [width*0.1, height], [width*0.9, height]])
            dst = np.float32([[width*0.2, 0], [width*0.8, 0],
                             [width*0.2, height], [width*0.8, height]])
            M = cv2.getPerspectiveTransform(src, dst)
            processed_img = cv2.warpPerspective(img, M, (width, height))
        elif style == "close_up":
            height, width = img.shape[:2]
            crop_size = min(height, width) // 2
            y_start = (height - crop_size) // 2
            x_start = (width - crop_size) // 2
            processed_img = img[y_start:y_start+crop_size, x_start:x_start+crop_size]
            processed_img = cv2.resize(processed_img, (width, height))
        
        results = model(processed_img, conf=conf)
        annotated_frame = results[0].plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frames.append(annotated_frame_rgb)
    
    imageio.mimsave(output_path, frames, fps=10)
    return output_path

def validate_label_format(label_path):
    """
    Validate YOLO label format and content
    
    Args:
        label_path (Path): Path to label file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            return True  # Empty files are valid (no objects)
            
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                return False
                
            class_id, x_center, y_center, width, height = map(float, parts)
            
            # Check if values are within valid range (0-1 for normalized coordinates)
            if not all(0 <= val <= 1 for val in [x_center, y_center, width, height]):
                return False
                
        return True
        
    except Exception:
        return False

def count_class_instances(labels_dir):
    """
    Count instances per class in labels directory
    
    Args:
        labels_dir (Path): Path to labels directory
        
    Returns:
        dict: Dictionary with class counts
    """
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # others, car, van, bus
    
    for label_file in Path(labels_dir).glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if class_id in class_counts:
                        class_counts[class_id] += 1
    
    return class_counts