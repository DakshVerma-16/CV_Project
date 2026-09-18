import os
import cv2
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.preprocessing import preprocess_image
from src.segmentation import segment_leaf
from src.feature_extraction import extract_features
from src.disease_detection import create_classifier, train_classifier


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "Plant_images_pianalytix"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "plant_disease_random_forest.pkl"
)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("=" * 60)
    print("LOADING PLANT DISEASE DATASET")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    # Discover class folders automatically
    class_names = sorted(
        [
            folder
            for folder in os.listdir(DATASET_PATH)
            if os.path.isdir(
                os.path.join(DATASET_PATH, folder)
            )
        ]
    )

    if len(class_names) == 0:
        raise ValueError(
            "No class folders found in the dataset."
        )

    print(f"Dataset path: {DATASET_PATH}")
    print(f"Classes found: {len(class_names)}")

    for class_name in class_names:
        print(f"  - {class_name}")

    features = []
    labels = []

    # --------------------------------------------------------
    # Process every class
    # --------------------------------------------------------

    for class_index, class_name in enumerate(class_names):

        class_path = os.path.join(
            DATASET_PATH,
            class_name
        )

        image_files = [
            file
            for file in os.listdir(class_path)
            if file.lower().endswith(IMAGE_EXTENSIONS)
        ]

        print()
        print(
            f"Processing class {class_index + 1}/"
            f"{len(class_names)}: {class_name}"
        )

        print(
            f"Images found: {len(image_files)}"
        )

        processed_count = 0

        for image_file in image_files:

            image_path = os.path.join(
                class_path,
                image_file
            )

            # Read image
            image = cv2.imread(image_path)

            if image is None:
                print(
                    f"WARNING: Could not read {image_file}"
                )
                continue

            try:

                # ------------------------------------------------
                # 1. PREPROCESSING
                # ------------------------------------------------

                processed_image = preprocess_image(
                    image
                )

                # ------------------------------------------------
                # 2. SEGMENTATION
                #
                # segment_leaf returns:
                # segmented_leaf
                # leaf_mask
                # disease_mask
                # ------------------------------------------------

                segmented_leaf, leaf_mask, disease_mask = (
                    segment_leaf(
                        processed_image
                    )
                )

                # ------------------------------------------------
                # 3. FEATURE EXTRACTION
                # ------------------------------------------------

                image_features = extract_features(
                    processed_image,
                    leaf_mask,
                    disease_mask
                )

                # Convert to 1-D NumPy array
                image_features = np.asarray(
                    image_features,
                    dtype=np.float32
                ).flatten()

                features.append(
                    image_features
                )

                labels.append(
                    class_index
                )

                processed_count += 1

            except Exception as e:

                print(
                    f"WARNING: Failed to process "
                    f"{image_file}: {e}"
                )

        print(
            f"Successfully processed: "
            f"{processed_count}/{len(image_files)}"
        )

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    if len(features) == 0:
        raise ValueError(
            "No features were extracted from the dataset."
        )

    # Make sure every image has the same number
    # of extracted features
    feature_lengths = [
        len(feature)
        for feature in features
    ]

    if len(set(feature_lengths)) != 1:

        raise ValueError(
            "Feature vectors have different lengths."
        )

    X = np.asarray(
        features,
        dtype=np.float32
    )

    y = np.asarray(
        labels,
        dtype=np.int32
    )

    print()
    print("=" * 60)
    print("DATASET LOADING COMPLETED")
    print("=" * 60)

    print(f"Total samples: {len(X)}")
    print(f"Feature vector size: {X.shape[1]}")
    print(f"Number of classes: {len(class_names)}")

    return X, y, class_names


# ============================================================
# TRAIN RANDOM FOREST MODEL
# ============================================================

def train_model():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    X, y, class_names = load_dataset()

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CREATING TRAIN / TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # --------------------------------------------------------
    # Create Random Forest classifier
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    model = create_classifier()

    model = train_classifier(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print()
    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_data = {
        "model": model,
        "class_names": class_names
    }

    joblib.dump(
        model_data,
        MODEL_PATH
    )

    print()
    print("=" * 60)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 60)

    print(f"Model path: {MODEL_PATH}")

    return model


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_model()