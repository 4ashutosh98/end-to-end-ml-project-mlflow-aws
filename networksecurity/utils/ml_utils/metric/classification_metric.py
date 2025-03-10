from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys

def get_classification_score(y_true, y_pred):
    try:
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)
        return ClassificationMetricArtifact(
            precision_score=model_precision_score,
            recall_score=model_recall_score,
            f1_score=model_f1_score
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e