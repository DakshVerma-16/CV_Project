# PlantDiseaseCV - Project Statement

## 1. Problem Statement

Plant diseases can reduce crop quality and productivity if they are not identified at an early stage. Manual identification of diseases from leaf symptoms can be time-consuming and may require expert knowledge.

PlantDiseaseCV is a computer vision based plant disease detection system that analyzes plant leaf images to identify selected diseases and estimate the affected leaf area.

The system uses image preprocessing, leaf segmentation, feature extraction, and a Random Forest machine learning classifier to perform disease detection. It also provides a severity level based on the percentage of the detected affected area.

---

## 2. Project Scope

The current version of PlantDiseaseCV focuses on detecting three plant disease classes:

1. Corn Common Rust
2. Potato Early Blight
3. Tomato Bacterial Spot

The system accepts a plant leaf image as command-line input and performs the complete analysis automatically.

The system provides:

- Image preprocessing
- Leaf segmentation
- Disease-region segmentation
- Computer vision feature extraction
- Machine learning based disease classification
- Prediction confidence
- Disease severity estimation
- Automatic text report generation

---

## 3. Target Users

The system can be useful for:

- Students learning Computer Vision and Machine Learning
- Researchers working on plant image analysis
- Agriculture-related project demonstrations
- Farmers or agricultural users as a future application after further validation

---

## 4. High-Level Features

### Image Processing

- Image loading
- Image resizing
- Noise reduction
- Contrast enhancement
- Image normalization

### Leaf Segmentation

- Green leaf region detection
- Background removal
- Disease-region segmentation

### Feature Extraction

- Color features
- Texture features
- Shape features
- Mask-based features

### Disease Detection

- Random Forest classifier
- Three disease classes
- Prediction confidence

### Severity Analysis

- Affected leaf area calculation
- Severity percentage
- Severity classification

### Report Generation

- Terminal report
- Automatic text report
- Prediction and severity summary

### Command-Line Execution

The complete system can be executed from the terminal using:

python -m src.main <image_path>

Example:

python -m src.main ".\data\Plant_images_pianalytix\Corn_(maize)___Common_rust_\RS_Rust 2729.JPG"

---

## 5. Project Workflow

Plant Leaf Image
        |
        v
Image Preprocessing
        |
        v
Leaf Segmentation
        |
        v
Disease Region Detection
        |
        v
Feature Extraction
        |
        v
Random Forest Classification
        |
        v
Disease Prediction
        |
        v
Severity Analysis
        |
        v
Report Generation

---

## 6. Expected Output

For each input image, the system provides:

- Predicted disease
- Prediction confidence
- Affected leaf area percentage
- Severity level
- Generated text report

---

## 7. Limitations

The current system supports only the three disease classes included in the training dataset.

The reported classification performance is based on the project's test dataset and should not be interpreted as a guarantee of performance on field images or unseen real-world conditions.

Severity estimation is based on image segmentation and detected disease-region pixels.

---

## 8. Future Scope

Possible future improvements include:

- Support for more plant species and diseases
- Deep learning based disease detection
- Mobile application integration
- Real-time camera-based detection
- Improved disease-region segmentation
- Larger and more diverse datasets
- Multilingual farmer-friendly reports
