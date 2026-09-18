# PlantDiseaseCV — Plant Disease Detection and Severity Analysis using Computer Vision

A command-line Computer Vision and Machine Learning system for identifying selected plant leaf diseases and estimating disease severity.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Random%20Forest-orange)
![Status](https://img.shields.io/badge/Status-Course%20Project-yellow)

---

## Table of Contents

* [Overview](#overview)
* [Problem Statement](#problem-statement)
* [Objectives](#objectives)
* [Features](#features)
* [Technologies Used](#technologies-used)
* [Project Architecture](#project-architecture)
* [Workflow](#workflow)
* [Dataset](#dataset)
* [Computer Vision Methodology](#computer-vision-methodology)

  * [Image Preprocessing](#image-preprocessing)
  * [Leaf and Disease Segmentation](#leaf-and-disease-segmentation)
  * [Feature Extraction](#feature-extraction)
* [Machine Learning Methodology](#machine-learning-methodology)
* [Model Evaluation](#model-evaluation)
* [Severity Analysis](#severity-analysis)
* [Report Generation](#report-generation)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Training the Model](#training-the-model)
* [Testing](#testing)
* [Sample Output](#sample-output)
* [Functional Requirements](#functional-requirements)
* [Non-Functional Requirements](#non-functional-requirements)
* [Limitations](#limitations)
* [Future Enhancements](#future-enhancements)
* [References](#references)
* [Author](#author)

---

## Overview

**PlantDiseaseCV** is a modular Computer Vision and Machine Learning system that analyzes plant leaf images to identify selected plant diseases and estimate disease severity.

The system processes an input leaf image through:

1. Image preprocessing
2. Leaf segmentation
3. Disease-region segmentation
4. Feature extraction
5. Random Forest classification
6. Disease prediction
7. Severity analysis
8. Automatic report generation

The complete pipeline can be executed from the command line.

---

## Problem Statement

Plant diseases can reduce crop quality and productivity when they are not identified early. Manual identification of diseases from leaf images can be time-consuming and may require expert knowledge.

PlantDiseaseCV automates disease identification from plant leaf images using image processing, Computer Vision techniques, feature extraction, and Machine Learning.

The system also estimates the percentage of the leaf area affected by detected disease regions and classifies severity into **Low, Moderate, High, or Severe**.

---

## Objectives

* Automate plant disease identification from leaf images.
* Quantify disease severity based on affected leaf area.
* Build a modular Computer Vision and Machine Learning pipeline.
* Provide prediction confidence for the detected disease.
* Generate readable analysis reports.
* Provide a command-line executable workflow suitable for project evaluation and testing.

---

## Features

* Command-line based end-to-end plant disease detection pipeline.
* Image resizing and preprocessing.
* Gaussian noise reduction.
* CLAHE-based contrast enhancement.
* HSV-based leaf segmentation.
* HSV-based disease-region segmentation.
* Morphological image processing.
* 25-dimensional feature extraction.
* Color, texture, shape, and mask-based features.
* Random Forest disease classification.
* Prediction confidence calculation.
* Disease severity percentage estimation.
* Severity-level classification.
* Automatic text report generation.
* Automated test suite covering core pipeline components.

---

## Technologies Used

| Technology      | Purpose                              |
| --------------- | ------------------------------------ |
| Python          | Core programming language            |
| OpenCV          | Image processing and Computer Vision |
| NumPy           | Numerical and array operations       |
| Scikit-learn    | Machine Learning and evaluation      |
| Joblib          | Model serialization                  |
| Python unittest | Automated testing                    |
| Git             | Version control                      |
| GitHub          | Source code repository               |

**Machine Learning Model:** Random Forest Classifier

---

## Project Architecture

The system follows a modular Computer Vision and Machine Learning pipeline.

```text
                  Input Plant Leaf Image
                           |
                           v
                  Image Preprocessing
                           |
                           v
                    Leaf Segmentation
                           |
                           v
               Disease Region Segmentation
                           |
                           v
                    Feature Extraction
                           |
                           v
                Random Forest Classifier
                           |
                           v
                    Disease Prediction
                           |
                 +---------+---------+
                 |                   |
                 v                   v
        Prediction Confidence   Severity Analysis
                                     |
                                     v
                            Severity Classification
                                     |
                                     v
                              Report Generation
```

---

## Workflow

```text
Load Image
    |
    v
Preprocess Image
    |
    v
Segment Leaf
    |
    v
Segment Disease Region
    |
    v
Extract 25 Features
    |
    v
Load Trained Model
    |
    v
Predict Disease
    |
    +----------------------+
    |                      |
    v                      v
Confidence            Severity Analysis
                           |
                           v
                    Generate Text Report
```

---

## Dataset

The current dataset contains **900 plant leaf images** across **3 disease classes**, with approximately 300 images per class.

### Supported Disease Classes

1. `Corn_(maize)___Common_rust_`
2. `Potato___Early_blight`
3. `Tomato___Bacterial_spot`

Dataset location:

```text
data/Plant_images_pianalytix/
```

The dataset is included in the repository.

---

# Computer Vision Methodology

## Image Preprocessing

Implemented in:

```text
src/preprocessing.py
```

The preprocessing pipeline includes:

* Image loading
* Image resizing
* Gaussian noise reduction
* HSV conversion
* CLAHE-based contrast enhancement
* Image normalization

Images are resized to:

```text
224 × 224 pixels
```

The purpose of preprocessing is to provide a consistent image representation before segmentation and feature extraction.

---

## Leaf and Disease Segmentation

Implemented in:

```text
src/segmentation.py
```

The segmentation module uses the **HSV color space** to identify leaf regions and possible disease regions.

### Leaf Segmentation

Green leaf regions are detected using HSV thresholding.

The segmentation process also uses:

* Morphological opening
* Morphological closing
* Noise removal
* Background suppression

### Disease-Region Segmentation

Possible yellow/brown disease regions are identified using HSV thresholding.

The disease mask is restricted to the detected leaf region.

### Affected Area Calculation

The system calculates the number of pixels in the leaf mask and disease mask to estimate the affected leaf area.

---

## Feature Extraction

Implemented in:

```text
src/feature_extraction.py
```

The system generates a **25-dimensional feature vector** for each processed image.

The features are grouped into:

### Color Features

Color statistics extracted from the image and leaf region.

### Texture Features

Image characteristics representing texture information.

### Shape Features

Shape-related characteristics calculated from the segmented leaf region.

### Mask Features

Features derived from the leaf and disease masks.

The resulting feature vector is passed to the Machine Learning classifier.

---

# Machine Learning Methodology

The disease classification module is implemented in:

```text
src/disease_detection.py
```

The project uses a:

```text
Random Forest Classifier
```

### Random Forest Configuration

```text
n_estimators = 150
max_depth = None
random_state = 42
n_jobs = -1
```

The classifier predicts one of the three supported disease classes.

The prediction also provides a confidence value based on the classifier's predicted class probabilities.

---

## Model Training

Model training is implemented in:

```text
src/train_model.py
```

The training workflow is:

```text
Dataset
   |
   v
Load Images
   |
   v
Preprocess Images
   |
   v
Segment Leaf
   |
   v
Extract 25 Features
   |
   v
Train/Test Split
   |
   v
Random Forest Training
   |
   v
Model Evaluation
   |
   v
Save Trained Model
```

### Dataset Split

```text
80% → Training
20% → Testing
```

The split uses:

```text
random_state = 42
stratify = y
```

The trained model is saved at:

```text
models/plant_disease_random_forest.pkl
```

The trained model is included in the repository.

---

# Model Evaluation

The trained Random Forest model achieved:

```text
Test Accuracy: 99.44%
```

The evaluation was performed on the project's **held-out test split of 180 images**.

### Classification Report

```text
                             precision    recall  f1-score   support

Corn_(maize)___Common_rust_       1.00      1.00      1.00        60
Potato___Early_blight             1.00      0.98      0.99        60
Tomato___Bacterial_spot           0.98      1.00      0.99        60

accuracy                           0.99       180
macro avg                          0.99      0.99      0.99       180
weighted avg                       0.99      0.99      0.99       180
```

### Confusion Matrix

```text
[[60  0  0]
 [ 0 59  1]
 [ 0  0 60]]
```

> **Important:** The 99.44% accuracy represents performance on the project's held-out test split only. It should not be interpreted as guaranteed real-world accuracy. Field images may contain different lighting conditions, backgrounds, camera characteristics, plant varieties, and disease appearances.

---

# Severity Analysis

Severity analysis is implemented in:

```text
src/severity_analysis.py
```

The system estimates the affected leaf area using:

```text
Severity (%) =
(Disease Area / Leaf Area) × 100
```

### Severity Classification

| Affected Leaf Area | Severity Level |
| -----------------: | -------------- |
|              0–10% | Low            |
|            >10–30% | Moderate       |
|            >30–60% | High           |
|               >60% | Severe         |

Severity estimation is based on the segmented disease-region mask and represents an image-based estimate.

---

# Report Generation

Report generation is implemented in:

```text
src/report_generator.py
```

The system automatically generates a text report containing:

* Input image name
* Analysis date and time
* Predicted disease
* Prediction confidence
* Affected leaf area
* Severity percentage
* Severity level
* Processing summary

Reports are saved in:

```text
outputs/
```

Example reports:

```text
outputs/RS_Rust 2729_report.txt
outputs/test_leaf_report.txt
```

---

# Project Structure

```text
PlantDiseaseCV/
│
├── data/
│   └── Plant_images_pianalytix/
│       ├── Corn_(maize)___Common_rust_/
│       ├── Potato___Early_blight/
│       └── Tomato___Bacterial_spot/
│
├── models/
│   └── plant_disease_random_forest.pkl
│
├── outputs/
│   ├── RS_Rust 2729_report.txt
│   └── test_leaf_report.txt
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── feature_extraction.py
│   ├── disease_detection.py
│   ├── severity_analysis.py
│   ├── report_generator.py
│   └── train_model.py
│
├── tests/
│   └── test_pipeline.py
│
├── README.md
├── requirements.txt
└── statement.md
```

---

# Installation

## 1. Clone the Repository

```powershell
git clone https://github.com/DakshVerma-16/CV_Project.git
cd PlantDiseaseCV
```

## 2. Create a Virtual Environment

On Windows:

```powershell
python -m venv venv
```

## 3. Activate the Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use the appropriate Python/PowerShell environment configuration for your system.

## 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

### Current Dependencies

```text
numpy==2.5.3
opencv-python==5.0.0.93
scikit-learn==1.9.1
joblib==1.6.0
```

---

# Usage

The complete pipeline can be executed from the command line.

## Run Disease Detection

```powershell
python -m src.main "<image_path>"
```

### Example

```powershell
python -m src.main ".\data\Plant_images_pianalytix\Corn_(maize)___Common_rust_\RS_Rust 2729.JPG"
```

The application performs:

```text
Load Image
    ↓
Preprocessing
    ↓
Leaf Segmentation
    ↓
Disease-Region Segmentation
    ↓
Feature Extraction
    ↓
Model Loading
    ↓
Disease Prediction
    ↓
Severity Analysis
    ↓
Report Generation
```

---

# Training the Model

To retrain the Random Forest model using the available dataset:

```powershell
python -m src.train_model
```

The training script:

1. Discovers the available disease classes.
2. Loads the dataset images.
3. Preprocesses each image.
4. Performs leaf and disease segmentation.
5. Extracts 25 features.
6. Splits the dataset into training and testing sets.
7. Trains the Random Forest classifier.
8. Evaluates the model.
9. Saves the trained model.

---

# Testing

The project includes **7 automated tests** in:

```text
tests/test_pipeline.py
```

The tests cover:

1. Image existence
2. Model existence
3. Image preprocessing
4. Leaf segmentation
5. Feature extraction
6. Model prediction
7. Severity analysis

Run the test suite using:

```powershell
python -m unittest discover -s tests -v
```

Expected result:

```text
Ran 7 tests

OK
```

---

# Sample Output

A successful test run on a Corn Common Rust image produced:

```text
============================================================
        PLANT DISEASE DETECTION REPORT
============================================================

Image Name        : RS_Rust 2729.JPG
Predicted Disease : Corn_(maize)___Common_rust_
Confidence        : 100.00%
Severity          : 40.38%
Severity Level    : High

============================================================
```

The generated report was saved at:

```text
outputs/RS_Rust 2729_report.txt
```

---

# Functional Requirements

The implemented system satisfies the following functional requirements:

1. Accept plant leaf images as input.
2. Preprocess input images.
3. Segment leaf regions.
4. Detect possible disease regions.
5. Extract Computer Vision features.
6. Classify diseases using Random Forest.
7. Provide prediction confidence.
8. Estimate affected leaf area.
9. Classify disease severity.
10. Generate a text report.
11. Execute the complete workflow from the command line.

---

# Non-Functional Requirements

## Maintainability

The project is divided into independent Python modules for preprocessing, segmentation, feature extraction, classification, severity analysis, reporting, and training.

## Usability

The complete pipeline can be executed using a simple command-line command.

## Reliability

The project includes validation checks and automated tests for important pipeline components.

## Error Handling

The main application checks whether the input image can be loaded and whether the trained model is available before continuing with analysis.

## Performance

The Random Forest classifier uses parallel processing:

```text
n_jobs = -1
```

## Scalability

The modular design allows additional disease classes, datasets, and improved models to be incorporated in future versions.

---

# Limitations

* Only three disease classes are currently supported.
* The current dataset contains 900 images.
* Severity estimation depends on the quality of image segmentation.
* The system is designed for the selected dataset and disease classes.
* Real-world field images may produce different results.
* The reported 99.44% accuracy applies only to the held-out test split.
* The project is a Computer Vision course prototype and should not be treated as a professional agricultural diagnostic system.

---

# Future Enhancements

Possible future improvements include:

* Support for more plant species and disease classes.
* Larger and more diverse datasets.
* Deep Learning models such as CNNs or transfer learning.
* Improved segmentation for complex backgrounds.
* Real-time camera-based detection.
* Web or mobile application integration.
* Visualization of detected disease regions.
* Improved severity estimation using advanced segmentation methods.
* Multilingual farmer-friendly reports.
* Explainable AI techniques for model predictions.

---

# References

* [OpenCV Documentation](https://docs.opencv.org/)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [NumPy Documentation](https://numpy.org/doc/)

---

# Author

**Daksh Verma**

GitHub: [DakshVerma-16](https://github.com/DakshVerma-16)

Repository: [PlantDiseaseCV](https://github.com/DakshVerma-16/CV_Project.git)

---

## Project Status

**Status: Course Project — Core Implementation Completed**

The current version includes:

* Computer Vision preprocessing
* Leaf and disease-region segmentation
* Feature extraction
* Random Forest classification
* Severity analysis
* Automatic report generation
* Command-line execution
* Automated testing
* GitHub version control

