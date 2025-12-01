# metrics.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import json
from typing import Dict, Any, List, Optional, Tuple

# ==============================================================================
# CORE METRICS CALCULATION
# ==============================================================================

def compute_classic_metrics(y_test, y_pred):
    """
    Compute and return basic evaluation metrics (accuracy, weighted F1-score, and report).

    Args:
        y_true: The true target labels (test set).
        y_pred: The predicted labels from the model.
        
    Returns:
        A dictionary containing "accuracy", "f1_score", and "report" (string).
    """

    accuracy = accuracy_score(y_test, y_pred)
    # Using 'weighted' average is good for classic models and general reporting, 
    # but remember to also check 'macro' or 'binary' F1 for your specific misinformation task.
    f1 = f1_score(y_test, y_pred, average="weighted") 
    report = classification_report(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "report": report
    }


# ==============================================================================
# DEEP LEARNING METRICS CALCULATION
# ==============================================================================
def compute_deep_metrics(p: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Computes metrics (F1-score and accuracy) from the prediction output 
    of a Hugging Face Trainer (TFTrainer or Trainer).

    Args:
        p: A tuple (predictions, label_ids) where:
           - p[0] is the numpy array of logits (raw model scores).
           - p[1] is the numpy array of true labels.
        
    Returns:
        A dictionary containing "f1_score" and "accuracy".
    """
    # p.predictions is the array of logits (raw scores)
    logits = p[0] 
    
    # Get the predicted class index (0 or 1) by finding the max logit across axis 1
    y_pred = np.argmax(logits, axis=1) 
    
    # The true labels are in p.label_ids or p[1]
    y_true = p[1]
    
    # Calculate F1-Score: Use 'macro' or 'weighted' average for balanced scoring, 
    # which is preferred for classification tasks like misinformation detection.
    f1_macro = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        'f1_score': f1_macro,
        'accuracy': accuracy,
    }


# ==============================================================================
# PLOTTING FUNCTION
# ==============================================================================

def plot_confusion_matrix(
    y_test: List[Any], 
    y_pred: List[Any], 
    labels: Optional[List[str]] = None, 
    figsize: tuple = (8, 6),
    title: str = "Confusion Matrix"
):
    """
    Plot a confusion matrix with the true/predicted labels.
    """

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    plt.figure(figsize=figsize)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
    plt.show()

# ==============================================================================
# SAVING FUNCTION
# ==============================================================================

def save_evaluation_report(report_dict: Dict[str, Any], filepath: str = "evaluation_report.json"):
    """
    Save evaluation results (accuracy, f1, report text) as JSON.
    
    Args:
        report_dict: The dictionary returned by compute_classic_metrics.
        filepath: The path to save the JSON file.
    """

    # We need to ensure we save the raw classification_report string
    output = {
        "accuracy": report_dict["accuracy"],
        "f1_score": report_dict["f1_score"],
        "classification_report": report_dict["report"]
    }

    with open(filepath, "w") as f:
        # Use json.dumps for the inner string representation of the classification report
        # We can't directly dump the classification_report string, so we ensure the entire output is serializable
        json.dump(output, f, indent=4)

    print(f"Evaluation report saved to {filepath}")
    
