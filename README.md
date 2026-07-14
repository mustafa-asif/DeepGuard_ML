# DeepGuard_ML

DeepGuard_ML is a deep learning–based project for **detecting deepfake face images (Real vs Fake)** using **TensorFlow**.  
The main workflow is implemented in the notebook: **`DeepGuard_pynb.ipynb`**.

---

## Project Overview

This project provides an end-to-end pipeline for:
- Setting up a TensorFlow environment (with GPU support when available)
- Downloading and preparing a deepfake dataset from Kaggle
- Building a labeled dataset (**real** vs **fake**)
- Training and evaluating an image classification model to detect deepfakes

---

## Notebook: `DeepGuard_pynb.ipynb`

The notebook covers:

### 1) Environment & Setup
- Prints **TensorFlow version** (example run shows **TensorFlow 2.19.0**)
- Checks **GPU availability** for faster training
- Configures **Kaggle credentials** to enable dataset download via Kaggle API

### 2) Dataset Download (Kaggle)
- Dataset used: **140k Real and Fake Faces** (`xhlulu/140k-real-and-fake-faces`)
- The notebook downloads and extracts the dataset, then organizes it into local directories.

### 3) Data Loading & Labeling
- Loads images from folder structure and assigns labels:
  - `real` → **real**
  - `fake` → **fake**
- Prints dataset distribution and basic stats.

**Example run output (from the notebook):**
- Total images loaded: **20,000**
  - Real: **10,000**
  - Fake: **10,000**

### 4) Visualization
- Produces inline plots/visual checks (e.g., sample previews and/or distribution charts).

---

## Dataset

- Source: Kaggle — **140k Real and Fake Faces**
- URL (as referenced in notebook): https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
- License: Listed on Kaggle (review before usage in production/commercial settings)

---

## Requirements

Typical requirements to run the notebook:
- Python 3.x
- TensorFlow (GPU optional but recommended)
- Kaggle API credentials configured (`kaggle.json`)
- Common ML/data libraries (NumPy, Pandas, Matplotlib, etc.)

> Tip: If you're running on Google Colab, make sure GPU is enabled:  
> **Runtime → Change runtime type → GPU**

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/mustafa-asif/DeepGuard_ML.git
   cd DeepGuard_ML
   ```

2. Open and run the notebook:
   - `DeepGuard_pynb.ipynb`

3. Ensure Kaggle credentials are available (one of the following):
   - Place `kaggle.json` in the expected location (commonly `~/.kaggle/kaggle.json`)
   - Or configure via your environment/Colab secrets as the notebook expects

---

## Repository Structure

- `DeepGuard_pynb.ipynb` — Main notebook: setup, dataset download, preprocessing, training/evaluation
- `README.md` — Project documentation

---
## Tech Stack

- Python
- XGBoost, scikit-learn (Random Forest, SVM, Logistic Regression)
- NumPy, Pandas
- OpenCV / scikit-image (for HOG, LBP, FFT feature extraction)
- Jupyter Notebook


# DeepGuard ML — Model Training Pipeline

Training pipeline for **DeepGuard AI**, an ensemble machine learning system that detects AI-generated / manipulated images. This repo covers dataset preparation, feature extraction, and model training — the trained models are consumed by the [DeepGuard AI web app](https://github.com/mustafa-asif/DeepGuard-AI-Detector).

## Overview

The pipeline trains and evaluates four ensemble classifiers to distinguish real images from AI-generated/forged ones, using classical feature extraction techniques rather than raw pixel input — keeping the models lightweight and interpretable.

## Models Trained

| Model               | Accuracy | F1-Score | Status  |
| -------------------- | -------- | -------- | ------- |
| XGBoost              | 94.5%    | 93.8%    | Primary |
| Random Forest         | 91.2%    | 90.5%    | Active  |
| SVM                   | 88.7%    | 87.9%    | Active  |
| Logistic Regression   | 86.4%    | 85.1%    | Active  |

## Feature Extraction

Rather than feeding raw images into the models, three complementary feature extraction techniques are used to capture different forgery signatures:

- **HOG (Histogram of Oriented Gradients)** — captures edge/structural anomalies
- **LBP (Local Binary Patterns)** — analyzes texture micro-patterns to detect GAN artifacts
- **FFT (Fast Fourier Transform)** — reveals frequency-domain irregularities and checkerboard artifacts common in generated images

## Dataset

- **Total size**: 140,000 images
- **Real images**: 70,000
- **AI-generated images**: 70,000
- Split into train/validation/test sets

## Pipeline

1. **Data loading** — download and organize the labeled real/fake image dataset
2. **Feature extraction** — compute HOG, LBP, and FFT features for each image
3. **Model training** — train XGBoost, Random Forest, SVM, and Logistic Regression classifiers on the extracted features
4. **Evaluation** — compare models on accuracy and F1-score
5. **Export** — serialize trained models (`ensemble.pkl` and individual classifier files) for use in the DeepGuard AI web app





## Related Repositories

- [DeepGuard AI Detector](https://github.com/mustafa-asif/DeepGuard-AI-Detector) — the deployed web app that uses these trained models



## Notes

- The notebook may prompt overwrite/replace actions when extracting files if they already exist locally.
- For reproducible runs, consider clearing/cleaning the dataset directory before re-extracting.

---

## Author

Mustafa Asif — [GitHub](https://github.com/mustafa-asif)
