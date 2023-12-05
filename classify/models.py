#!/usr/bin/env python3
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from .compute_metrics import compute_metrics
import time 


def classifier(X, y, X_test, y_test) -> SVC:
    """Train SVM classifier with X and y, evalute with X_test and y_test

    Args:
        X (n_samples, feature_dim): training features 
        y (n_samples,): training labels 
        
        X_test (n_samples, feature_dim): testing features 
        y_test (n_samples, ): testing labels
    
    Returns:
        clf: trained classifier 
        
    Example: 
        >>> import sys
        >>> sys.path.append("../")
        >>> from classify import classifier
        >>> classifier(X, y, X_test, y_test)
        Start training SVM.
        Training done. Time taken: 10.00 mins
        Evaluation on test set:
        {'uar': 0.720959595959596, 'f1': 0.628428927680798, 
        'macro_f1': 0.7195511855684803, 'accuracy': 0.7491582491582491}
    """
    svm = SVC(kernel="linear", random_state=2023)
    mlp = MLPClassifier(hidden_layer_sizes=(10,), 
                        random_state=2023,
                        max_iter=1000)
    
    for name, model in zip(["SVM", "MLP"], [svm, mlp]):
        
        print(f"---Start training {name}---")
        start = time.time()
        clf = make_pipeline(StandardScaler(), model)
        clf.fit(X, y)
        end = time.time()
        print(f"Training done. Time taken: {(end-start)/60:.2f} mins.")
        
        print("Evaluation on test set:")
        y_pred = clf.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        print(metrics)
    
    return clf 

