# Weapon & Threat Detection Using YOLO and Computer Vision
A comprehensive computer vision project for automatic detection and classification of threats in military and civilian environments, powered by YOLOv8/YOLOv9 object detection models. The system identifies objects such as weapons, soldiers, tanks, vehicles, civilians, and trenches. A user-friendly Streamlit web application is provided for easy interaction with the model for real-time detection, visualization, and analysis.

#Project structure

weapon-threat-detection/
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── yolov8/ (or yolov9/)
│   ├── runs/
│   └── best.pt
├── streamlit_app/
│   └── app.py
├── data.yaml
├── requirements.txt
└── README.md

# Project Goals
Detect multiple military and civilian objects:
soldiers, civilians, weapons, vehicles, tanks, trenches

Classify detected objects into Threat (e.g., weapons, enemy soldiers) and Non-Threat (e.g., civilians).

Provide real-time detection on images and videos.

Build an accessible Streamlit web app for end-users.

# Technologies Used
Purpose	Tool / Library
Object Detection	YOLOv8 / YOLOv9 (Ultralytics)
Image Processing	OpenCV, Pillow
Web App	Streamlit
Data Augmentation	Albumentations
Visualization	Matplotlib, Seaborn

# Dataset
Custom YOLO dataset organized as per YOLO format:

bash
Copy
Edit
train/images, train/labels  
val/images, val/labels  
test/images, test/labels
Classes:
['camouflage_soldier', 'weapon', 'military_tank', 'military_truck', 'military_vehicle', 'civilian', 'soldier', 'civilian_vehicle', 'trench']

Annotations follow the YOLO bounding box format:
<class_id> <x_center> <y_center> <width> <height> (normalized)


# Exploratory Data Analysis (EDA)
Image Analysis: size, resolution, aspect ratio, quality checks (blur, exposure)

Bounding Box Analysis: size distribution, aspect ratios

Class Distribution: balance check, underrepresented classes identified

Heatmaps: object density mapping

Visualization: sample images with bounding boxes for verification


# Challenges Identified
Class Imbalance: Small classes like trenches, weapons addressed via oversampling, targeted augmentation.

Small Object Detection: Improved using higher-resolution images and optimized YOLO anchors.

Annotation Quality: Verified and corrected using LabelImg / CVAT.

# 📈 Performance Evaluation
Metric	Result (Example)
Precision	88%
Recall	85%
F1 Score	86%
mAP@0.5	91%
Inference Time	~40 ms/image

# 🚨 Threat Classification Logic
Object Class	Category
Weapons, Tanks, Enemy Soldiers	Threat
Civilians, Friendly Soldiers	Non-Threat

# 🌐 Streamlit Web Application Features
Feature	Status
Upload Images / Videos	✅
Visualize YOLO Detections	✅
Threat / Non-Threat Classification	✅
Download Annotated Results	✅

📷 Image Detection Example:
Upload image → view YOLO boxes → classify as threat/non-threat → download result.

🎥 Video Detection Example:
Upload video → process frame-by-frame → download processed video with annotations.

# How to Run the Project
1️⃣ Install Requirements
pip install -r requirements.txt
2️⃣ Run Streamlit App
streamlit run app.py
3️⃣ Upload Files via Web Interface
Images (.jpg, .png)

Videos (.mp4, .avi)

# 📥 Outputs
Annotated images (bounding boxes + labels)

Processed videos with detections

Classification: Threat / Non-Threat

# 📌 Future Improvements
Deploy Streamlit app on Streamlit Cloud / Hugging Face Spaces

Add support for live webcam / CCTV feed

Incorporate alert notifications for detected threats

Refine class balancing with synthetic data (GANs, Diffusion models)

# 🤝 Credits
YOLO by Ultralytics

Albumentations for data augmentation

OpenCV & PIL for image processing

Streamlit for front-end interface

