import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from typing import Optional

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def train_model(X_train, y_train) -> BaseEstimator:
    """
    Train a logistic regression model with balanced class weights.

    Parameters:
    - X_train: Training features
    - y_train: Training labels

    Returns:
    - Trained model (sklearn estimator)
    """
    try:
        logging.info("Training Logistic Regression model with balanced class weight...")
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        logging.info("Model training complete.")
        return model
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise


def save_model(model: BaseEstimator, path: str = 'models/best_model.pkl') -> None:
    """
    Save the trained model to a file using joblib.

    Parameters:
    - model: Trained model
    - path: Destination file path
    """
    try:
        joblib.dump(model, path)
        logging.info(f"Model saved successfully at {path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        raise


def load_model(path: str = 'models/best_model.pkl') -> Optional[BaseEstimator]:
    """
    Load a trained model from a file.

    Parameters:
    - path: File path to the saved model

    Returns:
    - Loaded model or None if failed
    """
    try:
        model = joblib.load(path)
        logging.info(f"Model loaded successfully from {path}")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found at {path}")
        return None
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

