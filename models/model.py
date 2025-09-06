from ultralytics import YOLO
from config.config import Config

class YOLOModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def load_pretrained(self):
        """Load pretrained YOLO model"""
        self.model = YOLO(self.config.MODEL_NAME)
        return self.model
    
    def train(self):
        """Train the model"""
        if self.model is None:
            self.load_pretrained()
            
        results = self.model.train(
            data=f"{self.config.OUTPUT_DIR}/data.yaml",
            epochs=self.config.EPOCHS,
            imgsz=self.config.IMG_SIZE,
            batch=self.config.BATCH_SIZE,
            device='cuda',
            plots=True
        )
        
        return results
    
    def evaluate(self, model_path=None, conf=0.25):
        """Evaluate the model"""
        if model_path:
            self.model = YOLO(model_path)
        
        metrics = self.model.val(conf=conf)
        return metrics
    
    def predict(self, image_path, conf=0.25):
        """Make predictions on an image"""
        results = self.model(image_path, conf=conf)
        return results