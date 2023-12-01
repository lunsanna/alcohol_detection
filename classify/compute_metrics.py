#!/usr/bin/env python3
from sklearn.metrics import recall_score, accuracy_score, f1_score

def compute_metrics(y_true, y_pred):
    """_summary_

    Args:
        y_true (n_sample, feature_dim): features 
        y_pred (n_sample, ): labels 

    Returns:
        Dict: metrics dictionary 
    """
    
    uar = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    accuracy = accuracy_score(y_true, y_pred)
    
    return {"uar": uar, "f1": f1, "macro_f1": macro_f1, "accuracy": accuracy}