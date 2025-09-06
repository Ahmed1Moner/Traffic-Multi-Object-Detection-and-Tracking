from config.config import Config
from training.train import Trainer

def main():
    # Initialize configuration
    config = Config()
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Prepare dataset
    print("Preparing dataset...")
    trainer.prepare_data()
    
    # Train model
    print("Training model...")
    trainer.train_model()
    
    # Evaluate model
    print("Evaluating model...")
    model_path = "runs/detect/train10/weights/best.pt"  # Update with actual path
    metrics = trainer.evaluate_model(model_path)
    
    # Visualize results
    print("Visualizing results...")
    results_dir = "runs/detect/train10"
    val_images_dir = f"{config.OUTPUT_DIR}/images/val"
    trainer.visualize_results(results_dir, val_images_dir)

if __name__ == "__main__":
    main()