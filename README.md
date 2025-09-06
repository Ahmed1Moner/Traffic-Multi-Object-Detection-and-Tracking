<<<<<<< HEAD
# YOLOv8 Object Detection on UA-DETRAC Dataset

This project implements vehicle detection using YOLOv8 on the UA-DETRAC dataset, which contains challenging real-world traffic video sequences:cite[4].

## 📊 Dataset

The UA-DETRAC dataset contains approximately 10 hours of videos captured with a Canon EOS 550D camera at 24 different locations in Beijing and Tianjin, China. The dataset includes about 140,000 frames with rich annotations.

**Dataset Source**: [UA-DETRAC on Kaggle](https://www.kaggle.com/datasets/bratjay/ua-detrac-orig)

## 🏗️ Project Structure
yolov8-ua-detrac/
├── config/
│ ├── init.py
│ └── config.py
├── data/
│ ├── init.py
│ └── data_loader.py
├── models/
│ ├── init.py
│ └── model.py
├── training/
│ ├── init.py
│ └── train.py
├── utils/
│ ├── init.py
│ ├── visualization.py
│ └── helpers.py
├── plots/
├── runs/
├── requirements.txt
└── README.md


## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd yolov8-ua-detrac

Install dependencies:
pip install -r requirements.txt

Download the UA-DETRAC dataset from Kaggle and place it in the appropriate directory.


=======
# Traffic-Multi-Object-Detection-and-Tracking
>>>>>>> 1a373240e5ff9185b4200a1b349af81b8d88b778
