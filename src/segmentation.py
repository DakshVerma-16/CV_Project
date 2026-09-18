import cv2
import numpy as np


def create_leaf_mask(image):
    """
    Create a mask for the green leaf area using HSV color space.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV range for green leaf regions
    lower_green = np.array([25, 30, 20])
    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small gaps
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def create_disease_mask(image, leaf_mask):
    """
    Detect possible yellow/brown disease regions
    inside the leaf area.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Yellow/brown regions
    lower_disease = np.array([5, 40, 20])
    upper_disease = np.array([35, 255, 255])

    disease_mask = cv2.inRange(
        hsv,
        lower_disease,
        upper_disease
    )

    # Keep only regions inside the leaf
    disease_mask = cv2.bitwise_and(
        disease_mask,
        leaf_mask
    )

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)

    disease_mask = cv2.morphologyEx(
        disease_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    disease_mask = cv2.morphologyEx(
        disease_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return disease_mask


def apply_leaf_segmentation(image, leaf_mask):
    """
    Extract the leaf from the background.
    """

    segmented = cv2.bitwise_and(
        image,
        image,
        mask=leaf_mask
    )

    return segmented


def calculate_affected_area(
    leaf_mask,
    disease_mask
):
    """
    Calculate the percentage of leaf area
    affected by possible disease regions.
    """

    leaf_area = cv2.countNonZero(
        leaf_mask
    )

    disease_area = cv2.countNonZero(
        disease_mask
    )

    if leaf_area == 0:
        return 0.0

    percentage = (
        disease_area / leaf_area
    ) * 100

    return round(
        float(percentage),
        2
    )


def segment_leaf(image):
    """
    Complete segmentation pipeline.

    Returns:
        segmented_leaf
        leaf_mask
        disease_mask
    """

    leaf_mask = create_leaf_mask(
        image
    )

    disease_mask = create_disease_mask(
        image,
        leaf_mask
    )

    segmented_leaf = apply_leaf_segmentation(
        image,
        leaf_mask
    )

    return (
        segmented_leaf,
        leaf_mask,
        disease_mask
    )