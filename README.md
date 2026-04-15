# DualVision--Classification-Segmentation
Multi-task computer vision pipeline for multi-label classification and semantic segmentation with PCA and statistical validation (MLDS 2026 Assignment)
---

## 📌 Overview

This project implements two core vision tasks on natural images:

- **Multi-label classification**: Predict all object classes present in an image  
- **Semantic segmentation**: Predict pixel-wise class labels  

Additionally, the project includes:
- PCA implemented from scratch using NumPy  
- Statistical evaluation using Wilcoxon Signed-Rank Test and Bootstrap Confidence Intervals  

---

## 📂 Dataset

- **Train**: 2200 images (with labels and segmentation masks)  
- **Test**: 713 images (no ground truth provided)  

Each training sample includes:
- RGB image  
- Multi-label annotations (`labels.csv`)  
- Segmentation mask (palette-encoded PNG)  

---

## 🏗️ Project Structure
classification/ # Multi-label classification model
segmentation/ # Semantic segmentation model
pca/ # PCA implementation (NumPy)
utils/ # Metrics, losses, RLE encoding
notebooks/ # Training + statistical analysis
outputs/ # Logs, predictions, visualizations
submission/ # Kaggle submission file
report/ # Final report

---

## 🧠 Methodology

### 🔹 PCA (NumPy Implementation)

- Implemented using **Thin SVD**
- Mean-centered data matrix  
- Extracted principal components and eigenvalues  

Used for:
- Variance analysis  
- Reconstruction experiments  
- Dimensionality reduction  

---

### 🔹 Segmentation Model

- Custom **SegmentationBackbone + SegmentationModel**
- CNN encoder + learned decoder  
- Dice-based loss function  

Special handling:
- Pixels with label `255` are ignored during evaluation  

---

### 🔹 Classification Model

- CNN backbone (ImageNet-pretrained)  
- Custom classification head  
- Sigmoid activation for multi-label prediction  

---

## 📊 Evaluation Metrics

- **Classification**: Mean F1 Score  
- **Segmentation**: Mean IoU (mIoU)  

Final score:
```
Final Score = 0.5 × F1 + 0.5 × mIoU
```

---

## 📉 Statistical Testing
All statistical analysis is implemented in:
notebooks/statistical_tests.ipynb

### Wilcoxon Signed-Rank Test
- Compared two segmentation models  
- Evaluated significance of performance differences  

### Bootstrap Confidence Interval
- 1000 resamples  
- 95% confidence interval using percentile method  

---

## ⚙️ Installation

```bash
git clone https://github.com/sehajganjoo/DualVision--Classification-Segmentation.git
cd DualVision--Classification-Segmentation
pip install -r requirements.txt
```
---
## Training & Experiments

All training is performed in:
notebooks/training_notebook.ipynb

Includes:
- Data preprocessing
- Model training (classification + segmentation)
- Evaluation
- Visualization of results

---

## Inference

Inference logic is implemented within the training notebook and reused for test predictions.
---
## Submission Format
```
image_id, classification, segmentation_rle
```
- Classification: space-separated class names
- Segmentation: Run-Length Encoding (RLE)
---
## Constraints Followed
- Only ImageNet-pretrained backbones used
- No pre-trained segmentation models
- Custom architectures implemented
- No hardcoded predictions

---


