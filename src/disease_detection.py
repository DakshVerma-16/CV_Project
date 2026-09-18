import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def create_classifier():


    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    return model


def train_classifier(X_train, y_train):
    """
    Train the classifier using training features and labels.
    """

    model = create_classifier()

    print("Training Random Forest classifier...")
    model.fit(X_train, y_train)

    print("Training completed.")

    return model


def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate the trained classifier.
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    matrix = confusion_matrix(y_test, y_pred)

    return accuracy, report, matrix


def predict_disease(model, feature_vector):
    """
    Predict the disease from an extracted feature vector.
    """

    feature_vector = np.asarray(feature_vector).reshape(1, -1)

    prediction = model.predict(feature_vector)[0]

    confidence = 0.0

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_vector)[0]
        confidence = float(np.max(probabilities)) * 100

    return prediction, confidence


def save_model(model, model_path):
    """
    Save the trained model to a file.
    """

    joblib.dump(model, model_path)

    print(f"Model saved successfully: {model_path}")


def load_model(model_path):
    """
    Load a previously trained model.
    """

    model = joblib.load(model_path)

    print(f"Model loaded successfully: {model_path}")

    return model
