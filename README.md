# 🏛️ ArchiNet: Architectural Style Classifier

ArchiNet is a deep learning model that classifies images of buildings into different architectural styles — from Baroque and Bauhaus to Gothic and Greek Revival.

[![Open in Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/aalize/ArchiNet)

## Features
- Classifies images into **25 architectural styles**.
- Provides **top-3 predictions** with confidence scores.
- Supports both **CPU and GPU** inference.
- Optimized **ONNX model** for fast inference.

## Installation & Usage

### Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/aalizelau/ArchiNet.git
   cd ArchiNet
   
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Gradio app:
   ```bash
   python app.py
   ```

### Model Details
- **Input**: `384x384 RGB image` (ImageNet normalized)
- **Output**: Top-k probabilities across 25 styles
- **Format**: ONNX

## Dataset
The model was trained on the **Architecture Dataset** from Kaggle:  
🔗 [https://www.kaggle.com/datasets/wwymak/architecture-dataset](https://www.kaggle.com/datasets/wwymak/architecture-dataset)
