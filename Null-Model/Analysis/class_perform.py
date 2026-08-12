"""Count-based classification metrics.

Both arguments are 0/1 label vectors: ``yp`` the predicted class and ``yd`` the
observed one. Sensitivity and recall are the same quantity; both are returned
because the manuscript prints both rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Classification:
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    recall: float
    f1_score: float

    def as_dict(self) -> dict[str, float]:
        return {
            "Accuracy": self.accuracy,
            "Sensitivity": self.sensitivity,
            "Specificity": self.specificity,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1-score": self.f1_score,
        }


def class_perform(yp, yd) -> Classification:
    """Accuracy, sensitivity, specificity, precision, recall and F1."""
    yp = np.asarray(yp).astype(int).ravel()
    yd = np.asarray(yd).astype(int).ravel()
    if yp.shape != yd.shape:
        raise ValueError(f"shape mismatch: {yp.shape} against {yd.shape}")

    positives = int((yd == 1).sum())
    negatives = int((yd == 0).sum())
    true_positive = int(((yd == 1) & (yp == 1)).sum())
    true_negative = int(((yd == 0) & (yp == 0)).sum())
    predicted_positive = int((yp == 1).sum())

    sensitivity = true_positive / positives if positives else 0.0
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if precision + sensitivity else 0.0)

    return Classification(
        accuracy=float((yp == yd).mean()) if yd.size else 0.0,
        sensitivity=sensitivity,
        specificity=true_negative / negatives if negatives else 0.0,
        precision=precision,
        recall=sensitivity,
        f1_score=f1,
    )
