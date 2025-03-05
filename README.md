# Tennis Analysis System

A comprehensive machine learning and computer vision solution for analyzing tennis matches using YOLO, PyTorch, and keypoint extraction techniques.

## Overview

This project builds an end-to-end tennis analysis system that detects players, tracks the ball, identifies court keypoints, and analyzes gameplay. It combines state-of-the-art object detection, tracking, and custom keypoint extraction models to provide insightful analytics for tennis matches.
![image](https://github.com/user-attachments/assets/90289a92-acf0-4bd6-8fa1-11d6857cfaaa)

## Features

- **Object Detection**: Uses YOLOv8 to detect tennis players, rackets, and balls
- **Object Tracking**: Implements tracking algorithms to follow objects across video frames
- **Court Analysis**: Custom CNN to identify and extract court keypoints and lines
- **Player Movement Analysis**: Tracks and analyzes player positioning and movement patterns
- **Ball Trajectory Analysis**: Tracks the ball's path and calculates relevant statistics
- **Shot Classification**: Identifies different types of tennis shots
- **Performance Metrics**: Calculates statistics and performance indicators

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tennis-analysis.git
cd tennis-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
tennis-analysis/
├── data/
│   ├── court_keypoints/           # Labeled court keypoint data
│   └── custom_objects/            # Custom dataset for fine-tuning YOLO
├── models/
│   ├── court_keypoint_model.py    # CNN for court keypoint detection
│   └── trained_models/            # Saved trained models
├── src/
│   ├── detection.py               # Object detection with YOLO
│   ├── tracking.py                # Object tracking
│   ├── court_detection.py         # Court line and keypoint detection
│   ├── analysis.py                # Tennis play analysis
│   └── visualization.py           # Visualization utilities
├── main.py                        # Main application
└── requirements.txt               # Project dependencies
```

## Usage

### Basic Usage

1. Run analysis on a tennis video:
```bash
python main.py --input path/to/tennis_video.mp4 --output path/to/output_directory
```

2. Train court keypoint detector:
```bash
python train_keypoint_detector.py --data path/to/keypoint_dataset --epochs 50
```

3. Fine-tune YOLO for tennis-specific detection:
```bash
python finetune_yolo.py --data path/to/custom_dataset --epochs 100
```

### Configuration

Adjust parameters in `config.yaml` to customize:
- Detection confidence thresholds
- Tracking algorithm parameters
- Analysis metrics
- Visualization options

## Training Custom Models

### Court Keypoint Detector

1. Prepare labeled keypoint data in the format described in `data/README.md`
2. Run the training script:
```bash
python train_keypoint_detector.py --data data/court_keypoints --epochs 50
```

### Fine-tuning YOLO

1. Prepare labeled tennis object data in YOLO format
2. Run the fine-tuning script:
```bash
python finetune_yolo.py --data data/custom_objects --epochs 100
```

## Key Components

### Detection

Uses YOLOv8 to detect tennis-related objects including:
- Players
- Tennis balls
- Rackets
- Court lines and boundaries

### Tracking

Implements multiple tracking algorithms:
- SORT (Simple Online and Realtime Tracking)
- ByteTrack
- Custom tracking for tennis balls

### Court Keypoint Detection

Custom CNN to extract court keypoints:
- Court corners
- Service boxes
- Baselines
- Net position

### Analysis

Extracts gameplay metrics:
- Player positioning and movement patterns
- Ball trajectory analysis
- Shot classification
- Rally statistics
- Court coverage


