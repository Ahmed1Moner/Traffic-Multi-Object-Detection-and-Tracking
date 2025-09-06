import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

class Visualizer:
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_training_curves(self, results_dir):
        """Plot training curves from results.csv"""
        results_csv = Path(results_dir) / 'results.csv'
        
        if results_csv.exists():
            results_df = pd.read_csv(results_csv)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Box loss
            ax1.plot(results_df['epoch'], results_df['train/box_loss'], label='Train Box Loss', linewidth=2)
            ax1.plot(results_df['epoch'], results_df['val/box_loss'], label='Val Box Loss', linewidth=2)
            ax1.set_title('Box Loss', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Class loss
            ax2.plot(results_df['epoch'], results_df['train/cls_loss'], label='Train Class Loss', linewidth=2)
            ax2.plot(results_df['epoch'], results_df['val/cls_loss'], label='Val Class Loss', linewidth=2)
            ax2.set_title('Class Loss', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # DFL loss
            ax3.plot(results_df['epoch'], results_df['train/dfl_loss'], label='Train DFL Loss', linewidth=2)
            ax3.plot(results_df['epoch'], results_df['val/dfl_loss'], label='Val DFL Loss', linewidth=2)
            ax3.set_title('DFL Loss', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # mAP50
            ax4.plot(results_df['epoch'], results_df['metrics/mAP50(B)'], label='mAP50', linewidth=2, color='green')
            ax4.set_title('mAP50', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('mAP50')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(Path(results_dir) / 'training_curves.png')
            plt.show()
    
    def plot_metrics(self):
        """Plot performance metrics"""
        metrics_data = {
            'Class': ['All', 'Others', 'Car', 'Van', 'Bus'],
            'Precision': [0.296, 0.272, 0.456, 0.0944, 0.363],
            'Recall': [0.545, 0.285, 0.811, 0.279, 0.805],
            'mAP50': [0.447, 0.367, 0.718, 0.0718, 0.631],
            'mAP50-95': [0.286, 0.264, 0.498, 0.0437, 0.339]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        print("\n=== MODEL EVALUATION METRICS ===")
        print(metrics_df.to_string(index=False))
        
        # Create visual metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        classes = ['Others', 'Car', 'Van', 'Bus']
        
        # Precision by class
        precision = [0.272, 0.456, 0.0944, 0.363]
        axes[0, 0].bar(classes, precision, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[0, 0].set_title('Precision by Class', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Precision')
        for i, v in enumerate(precision):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Recall by class
        recall = [0.285, 0.811, 0.279, 0.805]
        axes[0, 1].bar(classes, recall, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[0, 1].set_title('Recall by Class', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Recall')
        for i, v in enumerate(recall):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # mAP50 by class
        map50 = [0.367, 0.718, 0.0718, 0.631]
        axes[1, 0].bar(classes, map50, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[1, 0].set_title('mAP50 by Class', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('mAP50')
        for i, v in enumerate(map50):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # mAP50-95 by class
        map5095 = [0.264, 0.498, 0.0437, 0.339]
        axes[1, 1].bar(classes, map5095, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[1, 1].set_title('mAP50-95 by Class', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('mAP50-95')
        for i, v in enumerate(map5095):
            axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('metrics_comparison.png')
        plt.show()
    
    def plot_sample_predictions(self, model, val_images_dir, num_samples=3):
        """Plot sample predictions on validation images"""
        val_images = list(Path(val_images_dir).glob('*.jpg'))[:num_samples]
        
        for img_path in val_images:
            results = model(img_path, conf=0.25)
            
            if len(results[0].boxes) > 0:
                plotted = results[0].plot()
                plt.figure(figsize=(12, 8))
                plt.imshow(plotted[:, :, ::-1])
                plt.title(f"Predictions: {img_path.name}", fontsize=14, fontweight='bold')
                plt.axis('off')
                plt.savefig(f"prediction_{img_path.name}.png")
                plt.show()
                
                # Print detection details
                print(f"\nDetections in {img_path.name}:")
                detections_by_class = {'others': 0, 'car': 0, 'van': 0, 'bus': 0}
                
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    detections_by_class[class_name] += 1
                    
                for cls, count in detections_by_class.items():
                    print(f"  {cls}: {count} detections")
            
            print("-" * 50)