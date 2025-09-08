# 🚗 Traffic Multi-Object Detection and Tracking with YOLOv8

This project implements **multi-object vehicle detection and tracking** using [YOLOv8](https://github.com/ultralytics/ultralytics) on the [UA-DETRAC dataset](https://www.kaggle.com/datasets/bratjay/ua-detrac-orig).  
The UA-DETRAC benchmark provides a challenging real-world traffic surveillance dataset captured in multiple urban scenarios, making it suitable for testing object detection and tracking models.

---

## 📊 Dataset: UA-DETRAC

The **UA-DETRAC dataset** is a large-scale benchmark for object detection and tracking in traffic surveillance. It includes:

- ~10 hours of video footage recorded at **24 different locations** in Beijing and Tianjin, China.
- Captured using a **Canon EOS 550D camera**.
- ~140,000 manually annotated frames.
- Four main vehicle categories: **car, bus, van, and others**.
- Challenging conditions such as **occlusion, weather variation, and illumination changes**.

👉 [UA-DETRAC Dataset on Kaggle](https://www.kaggle.com/datasets/bratjay/ua-detrac-orig)

---

## 🚀 Quick Start

1. Clone Repository
2. Install Dependencies
3. Download Dataset
4. Train the Model
5. Evaluate / Inference


Results

Training Metrics: Precision, Recall, mAP (mean Average Precision).
Tracking Performance: Multiple Object Tracking Accuracy (MOTA) and ID switches.
Example detection results are saved under plots/ and runs/.

Requirements

Python 3.8+
PyTorch (>=1.8)
YOLOv8 (Ultralytics)
Other dependencies:
