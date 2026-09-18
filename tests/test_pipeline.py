import os
import unittest

import cv2
import numpy as np

from src.preprocessing import preprocess_image
from src.segmentation import segment_leaf
from src.feature_extraction import extract_features
from src.disease_detection import load_model, predict_disease
from src.severity_analysis import analyze_severity


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "plant_disease_random_forest.pkl"
)

IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "Plant_images_pianalytix",
    "Corn_(maize)___Common_rust_",
    "RS_Rust 2729.JPG"
)


class TestPlantDiseasePipeline(unittest.TestCase):

    def test_image_exists(self):
        """Check that the test image exists."""
        self.assertTrue(
            os.path.exists(IMAGE_PATH)
        )

    def test_model_exists(self):
        """Check that the trained model exists."""
        self.assertTrue(
            os.path.exists(MODEL_PATH)
        )

    def test_preprocessing(self):
        """Check image preprocessing output."""
        image = cv2.imread(
            IMAGE_PATH
        )

        self.assertIsNotNone(image)

        processed = preprocess_image(
            image
        )

        self.assertEqual(
            processed.shape,
            (224, 224, 3)
        )

    def test_segmentation(self):
        """Check leaf and disease segmentation."""
        image = cv2.imread(
            IMAGE_PATH
        )

        processed = preprocess_image(
            image
        )

        (
            segmented_leaf,
            leaf_mask,
            disease_mask
        ) = segment_leaf(
            processed
        )

        self.assertEqual(
            segmented_leaf.shape,
            (224, 224, 3)
        )

        self.assertEqual(
            leaf_mask.shape,
            (224, 224)
        )

        self.assertEqual(
            disease_mask.shape,
            (224, 224)
        )

    def test_feature_extraction(self):
        """Check that 25 features are extracted."""
        image = cv2.imread(
            IMAGE_PATH
        )

        processed = preprocess_image(
            image
        )

        (
            segmented_leaf,
            leaf_mask,
            disease_mask
        ) = segment_leaf(
            processed
        )

        features = extract_features(
            processed,
            leaf_mask,
            disease_mask
        )

        self.assertEqual(
            len(features),
            25
        )

    def test_model_prediction(self):
        """Check trained model prediction."""
        image = cv2.imread(
            IMAGE_PATH
        )

        processed = preprocess_image(
            image
        )

        (
            segmented_leaf,
            leaf_mask,
            disease_mask
        ) = segment_leaf(
            processed
        )

        features = extract_features(
            processed,
            leaf_mask,
            disease_mask
        )

        model_data = load_model(
            MODEL_PATH
        )

        model = model_data["model"]
        class_names = model_data["class_names"]

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

        self.assertGreaterEqual(
            class_index,
            0
        )

        self.assertLess(
            class_index,
            len(class_names)
        )

        self.assertGreaterEqual(
            confidence,
            0.0
        )

        self.assertLessEqual(
            confidence,
            100.0
        )

    def test_severity_analysis(self):
        """Check severity analysis output."""
        image = cv2.imread(
            IMAGE_PATH
        )

        processed = preprocess_image(
            image
        )

        (
            segmented_leaf,
            leaf_mask,
            disease_mask
        ) = segment_leaf(
            processed
        )

        result = analyze_severity(
            leaf_mask,
            disease_mask
        )

        self.assertIn(
            "severity_percentage",
            result
        )

        self.assertIn(
            "severity_level",
            result
        )

        self.assertGreaterEqual(
            result["severity_percentage"],
            0.0
        )

        self.assertLessEqual(
            result["severity_percentage"],
            100.0
        )

        self.assertIn(
            result["severity_level"],
            [
                "Low",
                "Moderate",
                "High",
                "Severe"
            ]
        )


if __name__ == "__main__":
    unittest.main()