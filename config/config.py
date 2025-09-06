class Config:
    # Dataset paths
    BASE_DIR = "/home/yazan/ua-detrac"
    IMAGE_DIR = f"{BASE_DIR}/DETRAC-Images/DETRAC-Images"
    TRAIN_ANN_DIR = f"{BASE_DIR}/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML"
    TEST_ANN_DIR = f"{BASE_DIR}/DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML"
    
    # Training parameters
    IMG_SIZE = 640
    BATCH_SIZE = 16
    EPOCHS = 50
    CONFIDENCE_THRESHOLD = 0.25
    
    # Model
    MODEL_NAME = "yolov8n.pt"
    
    # Output
    OUTPUT_DIR = "/home/yazan/proper_dataset"
    
    # Classes
    CLASS_NAMES = {0: 'others', 1: 'car', 2: 'van', 3: 'bus'}