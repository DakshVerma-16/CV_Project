import os
import sys
import cv2

from src.preprocessing import preprocess_image
from src.segmentation import segment_leaf
from src.feature_extraction import extract_features
from src.disease_detection import load_model, predict_disease
from src.severity_analysis import analyze_severity
from src.report_generator import generate_report, print_report


# Project paths
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "plant_disease_random_forest.pkl"
)

OUTPUT_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "outputs"
)


def run_pipeline(image_path):
    """
    Run the complete plant disease detection pipeline.

    Pipeline:
        Image
        -> Preprocessing
        -> Segmentation
        -> Feature Extraction
        -> Disease Detection
        -> Severity Analysis
        -> Report Generation
    """

    print()
    print("=" * 60)
    print("        PLANT DISEASE DETECTION SYSTEM")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Validate image path
    # ---------------------------------------------------------

    if not os.path.exists(image_path):
        print()
        print("ERROR: Image file not found.")
        print(f"Image path: {image_path}")
        return False

    print()
    print(f"Input image: {image_path}")

    # ---------------------------------------------------------
    # 2. Load image
    # ---------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:
        print()
        print("ERROR: Unable to read the image.")
        return False

    print("Image loaded successfully.")
    print(f"Original image size: {image.shape}")

    # ---------------------------------------------------------
    # 3. Preprocessing
    # ---------------------------------------------------------

    print()
    print("Step 1: Preprocessing image...")

    processed_image = preprocess_image(
        image
    )

    print(
        f"Preprocessed image size: "
        f"{processed_image.shape}"
    )

    # ---------------------------------------------------------
    # 4. Segmentation
    # ---------------------------------------------------------

    print()
    print("Step 2: Segmenting leaf and disease regions...")

    (
        segmented_leaf,
        leaf_mask,
        disease_mask
    ) = segment_leaf(
        processed_image
    )

    print("Leaf segmentation completed.")
    print("Disease-region segmentation completed.")

    # ---------------------------------------------------------
    # 5. Feature extraction
    # ---------------------------------------------------------

    print()
    print("Step 3: Extracting features...")

    features = extract_features(
        processed_image,
        leaf_mask,
        disease_mask
    )

    print(
        f"Feature vector size: "
        f"{len(features)}"
    )

    # ---------------------------------------------------------
    # 6. Load trained model
    # ---------------------------------------------------------

    print()
    print("Step 4: Loading trained model...")

    if not os.path.exists(MODEL_PATH):
        print()
        print("ERROR: Trained model not found.")
        print(f"Expected model: {MODEL_PATH}")
        print()
        print(
            "Run this command first:"
        )
        print(
            "python -m src.train_model"
        )
        return False

    model_data = load_model(
        MODEL_PATH
    )

    model = model_data["model"]
    class_names = model_data["class_names"]

    print("Model loaded successfully.")
    print(f"Classes: {class_names}")

    # ---------------------------------------------------------
    # 7. Disease prediction
    # ---------------------------------------------------------

    print()
    print("Step 5: Detecting plant disease...")

    prediction = predict_disease(
        model,
        features
    )

    class_index = int(
        prediction[0]
    )

    confidence = float(
        prediction[1]
    )

    # Convert class index into disease name
    if (
        class_index < 0
        or class_index >= len(class_names)
    ):
        print()
        print("ERROR: Invalid predicted class index.")
        return False

    predicted_disease = class_names[
        class_index
    ]

    print(
        f"Predicted disease: "
        f"{predicted_disease}"
    )

    print(
        f"Prediction confidence: "
        f"{confidence:.2f}%"
    )

    # ---------------------------------------------------------
    # 8. Severity analysis
    # ---------------------------------------------------------

    print()
    print("Step 6: Analyzing disease severity...")

    severity_result = analyze_severity(
        leaf_mask,
        disease_mask
    )

    severity_percentage = (
        severity_result[
            "severity_percentage"
        ]
    )

    severity_level = (
        severity_result[
            "severity_level"
        ]
    )

    print(
        f"Affected leaf area: "
        f"{severity_percentage:.2f}%"
    )

    print(
        f"Severity level: "
        f"{severity_level}"
    )

    # ---------------------------------------------------------
    # 9. Display final report
    # ---------------------------------------------------------

    image_name = os.path.basename(
        image_path
    )

    print_report(
        image_name,
        predicted_disease,
        confidence,
        severity_percentage,
        severity_level
    )

    # ---------------------------------------------------------
    # 10. Save report
    # ---------------------------------------------------------

    print()
    print("Step 7: Generating report...")

    report_path = generate_report(
        image_name=image_name,
        predicted_disease=predicted_disease,
        confidence=confidence,
        severity_percentage=severity_percentage,
        severity_level=severity_level,
        output_directory=OUTPUT_DIRECTORY
    )

    print()
    print(f"Report saved at: {report_path}")

    print()
    print("=" * 60)
    print("          PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return True


def main():
    """
    Command-line entry point.
    """

    # Image path must be provided as a command-line argument
    if len(sys.argv) != 2:
        print()
        print("Usage:")
        print(
            "python -m src.main <image_path>"
        )
        print()
        print("Example:")
        print(
            "python -m src.main "
            "\"data\\Plant_images_pianalytix\\"
            "Corn_(maize)___Common_rust_\\"
            "RS_Rust 2469.JPG\""
        )
        return

    image_path = sys.argv[1]

    success = run_pipeline(
        image_path
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()