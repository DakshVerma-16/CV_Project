import cv2
import numpy as np


def extract_color_features(image):
    """
    Extract color-based features from the image.

    Features:
    - Mean BGR values
    - Standard deviation of BGR values
    - Mean HSV values
    - Standard deviation of HSV values
    """

    # BGR features
    bgr_mean = np.mean(
        image,
        axis=(0, 1)
    )

    bgr_std = np.std(
        image,
        axis=(0, 1)
    )

    # HSV features
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    hsv_mean = np.mean(
        hsv,
        axis=(0, 1)
    )

    hsv_std = np.std(
        hsv,
        axis=(0, 1)
    )

    return np.concatenate(
        [
            bgr_mean,
            bgr_std,
            hsv_mean,
            hsv_std
        ]
    )


def extract_texture_features(image):
    """
    Extract simple texture/statistical features.

    Features:
    - Mean grayscale intensity
    - Standard deviation
    - Minimum intensity
    - Maximum intensity
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    features = np.array(
        [
            np.mean(gray),
            np.std(gray),
            np.min(gray),
            np.max(gray)
        ],
        dtype=np.float32
    )

    return features


def extract_shape_features(leaf_mask):
    """
    Extract shape-based features from the leaf mask.

    Features:
    - Leaf area
    - Perimeter
    - Width
    - Height
    - Aspect ratio
    - Extent
    """

    contours, _ = cv2.findContours(
        leaf_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.zeros(
            6,
            dtype=np.float32
        )

    # Select largest contour
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(
        largest_contour
    )

    perimeter = cv2.arcLength(
        largest_contour,
        True
    )

    x, y, width, height = cv2.boundingRect(
        largest_contour
    )

    if height == 0:
        aspect_ratio = 0.0
    else:
        aspect_ratio = width / height

    bounding_area = width * height

    if bounding_area == 0:
        extent = 0.0
    else:
        extent = area / bounding_area

    return np.array(
        [
            area,
            perimeter,
            width,
            height,
            aspect_ratio,
            extent
        ],
        dtype=np.float32
    )


def extract_mask_features(
    leaf_mask,
    disease_mask
):
    """
    Extract features from leaf and disease masks.

    Features:
    - Leaf area
    - Disease area
    - Disease percentage
    """

    leaf_area = cv2.countNonZero(
        leaf_mask
    )

    disease_area = cv2.countNonZero(
        disease_mask
    )

    if leaf_area == 0:
        disease_percentage = 0.0
    else:
        disease_percentage = (
            disease_area / leaf_area
        ) * 100

    return np.array(
        [
            leaf_area,
            disease_area,
            disease_percentage
        ],
        dtype=np.float32
    )


def extract_features(
    image,
    leaf_mask,
    disease_mask
):
    """
    Extract the complete feature vector.

    Combines:
    1. Color features
    2. Texture features
    3. Shape features
    4. Mask/affected-area features
    """

    color_features = extract_color_features(
        image
    )

    texture_features = extract_texture_features(
        image
    )

    shape_features = extract_shape_features(
        leaf_mask
    )

    mask_features = extract_mask_features(
        leaf_mask,
        disease_mask
    )

    feature_vector = np.concatenate(
        [
            color_features,
            texture_features,
            shape_features,
            mask_features
        ]
    )

    return feature_vector.astype(
        np.float32
    )