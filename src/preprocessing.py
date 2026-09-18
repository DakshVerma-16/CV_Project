import cv2
import numpy as np


def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    return image


def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size)


def denoise_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(l_channel)

    enhanced = cv2.merge(
        (l_channel, a_channel, b_channel)
    )

    return cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )


def normalize_image(image):
    return image.astype(np.float32) / 255.0


def preprocess_image(image):
    image = resize_image(image)
    image = denoise_image(image)
    image = enhance_contrast(image)

    return image