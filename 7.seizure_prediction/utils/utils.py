import numpy as np
def calculate_metrics(conf_matrix):
    """
    Calculate accuracy, TPR, FPR, TNR, FNR, F1, and F2 from a confusion matrix.
    Args:
        conf_matrix: Confusion matrix (3x3 for 3 classes).
    Returns:
        Dictionary of metrics.
    """
    metrics = {}
    
    # True Positives (diagonal of the confusion matrix)
    TP = np.diag(conf_matrix)
    
    # False Positives (sum of columns minus diagonal)
    FP = np.sum(conf_matrix, axis=0) - TP
    
    # False Negatives (sum of rows minus diagonal)
    FN = np.sum(conf_matrix, axis=1) - TP
    
    # True Negatives (total samples minus TP, FP, FN)
    TN = np.sum(conf_matrix) - (TP + FP + FN)
    
    # Accuracy
    metrics["accuracy"] = np.sum(TP) / np.sum(conf_matrix)
    
    # True Positive Rate (Recall)
    metrics["TPR"] = np.divide(TP, TP + FN, where=(TP + FN) != 0)
    
    # False Positive Rate
    metrics["FPR"] = np.divide(FP, FP + TN, where=(FP + TN) != 0)
    
    # True Negative Rate
    metrics["TNR"] = np.divide(TN, TN + FP, where=(TN + FP) != 0)
    
    # False Negative Rate
    metrics["FNR"] = np.divide(FN, TP + FN, where=(TP + FN) != 0)
    
    # Precision
    precision = np.divide(TP, TP + FP, where=(TP + FP) != 0)
    
    # F1 Score
    metrics["F1"] = np.divide(2 * (precision * metrics["TPR"]), (precision + metrics["TPR"]), where=(precision + metrics["TPR"]) != 0)
    
    # F2 Score
    metrics["F2"] = np.divide(5 * (precision * metrics["TPR"]), (4 * precision + metrics["TPR"]), where=(4 * precision + metrics["TPR"]) != 0)
    
    return metrics
