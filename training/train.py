from models.model import YOLOModel
from data.data_loader import DataLoader
from config.config import Config
from utils.visualization import Visualizer

class Trainer:
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.model = YOLOModel(config)
        self.visualizer = Visualizer()
    
    def prepare_data(self):
        """Prepare the dataset"""
        self.data_loader.prepare_dataset()
    
    def train_model(self):
        """Train the model"""
        return self.model.train()
    
    def evaluate_model(self, model_path):
        """Evaluate the trained model"""
        return self.model.evaluate(model_path)
    
    def visualize_results(self, results_dir, val_images_dir):
        """Visualize training results and predictions"""
        self.visualizer.plot_training_curves(results_dir)
        self.visualizer.plot_metrics()
        self.visualizer.plot_sample_predictions(self.model.model, val_images_dir)