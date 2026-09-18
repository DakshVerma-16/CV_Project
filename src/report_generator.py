import os
from datetime import datetime


def generate_report(
    image_name,
    predicted_disease,
    confidence,
    severity_percentage,
    severity_level,
    output_directory="outputs"
):
    """
    Generate a text report containing the
    plant disease detection results.
    """

    # Create output directory if it does not exist
    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # Create a safe filename
    base_name = os.path.splitext(
        os.path.basename(image_name)
    )[0]

    report_path = os.path.join(
        output_directory,
        f"{base_name}_report.txt"
    )

    # Generate report
    report = f"""
============================================================
             PLANT DISEASE DETECTION REPORT
============================================================

Image Name       : {image_name}
Analysis Date    : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

------------------------------------------------------------
DISEASE DETECTION
------------------------------------------------------------

Predicted Disease : {predicted_disease}
Confidence        : {confidence:.2f}%

------------------------------------------------------------
SEVERITY ANALYSIS
------------------------------------------------------------

Affected Leaf Area : {severity_percentage:.2f}%
Severity Level     : {severity_level}

------------------------------------------------------------
SUMMARY
------------------------------------------------------------

The plant leaf image was processed using:
1. Image preprocessing
2. Leaf segmentation
3. Disease-region segmentation
4. Feature extraction
5. Random Forest classification
6. Severity analysis

============================================================
"""

    # Save report
    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report.strip()
        )

    return report_path


def print_report(
    image_name,
    predicted_disease,
    confidence,
    severity_percentage,
    severity_level
):
    """
    Print the detection results directly
    in the terminal.
    """

    print()
    print("=" * 60)
    print("PLANT DISEASE DETECTION REPORT")
    print("=" * 60)

    print(f"Image Name        : {image_name}")
    print(f"Predicted Disease : {predicted_disease}")
    print(f"Confidence        : {confidence:.2f}%")
    print(
        f"Severity          : "
        f"{severity_percentage:.2f}%"
    )
    print(
        f"Severity Level    : "
        f"{severity_level}"
    )

    print("=" * 60)