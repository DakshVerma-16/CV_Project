import cv2


def calculate_severity_percentage(leaf_mask, disease_mask):
    """
    Calculate the percentage of leaf area affected
    by possible disease regions.

    Formula:
        Severity (%) =
        (Disease Area / Leaf Area) * 100
    """

    leaf_area = cv2.countNonZero(leaf_mask)
    disease_area = cv2.countNonZero(disease_mask)

    if leaf_area == 0:
        return 0.0

    severity = (disease_area / leaf_area) * 100

    return round(float(severity), 2)


def classify_severity(severity_percentage):
    """
    Convert the severity percentage into a severity level.

    Levels:
        0-10%   -> Low
        10-30%  -> Moderate
        30-60%  -> High
        >60%    -> Severe
    """

    if severity_percentage <= 10:
        return "Low"

    elif severity_percentage <= 30:
        return "Moderate"

    elif severity_percentage <= 60:
        return "High"

    else:
        return "Severe"


def analyze_severity(leaf_mask, disease_mask):
    """
    Complete severity analysis.

    Returns:
        severity_percentage
        severity_level
    """

    severity_percentage = calculate_severity_percentage(
        leaf_mask,
        disease_mask
    )

    severity_level = classify_severity(
        severity_percentage
    )

    return {
        "severity_percentage": severity_percentage,
        "severity_level": severity_level
    }