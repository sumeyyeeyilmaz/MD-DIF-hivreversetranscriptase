"""Tanimoto similarity between two fingerprint vectors.

    S = dot(X,Y) / (dot(X,X) + dot(Y,Y) - dot(X,Y))

For binary vectors this is the intersection over the union, so it measures how
much of the combined substructure of two compounds they share.
"""

from __future__ import annotations

import numpy as np


def tan_sim(x, y) -> float:
    """Similarity of two fingerprints, 0 for disjoint and 1 for identical."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    denominator = x @ x + y @ y - x @ y
    return float(x @ y / denominator) if denominator else 0.0


def nearest_training_similarity(chembl: np.ndarray,
                                drugs: np.ndarray) -> np.ndarray:
    """For each ChEMBL compound, its similarity to the closest training RTI.

    The row maximum of the compound by training-panel similarity matrix. It
    describes how far the external set sits from the ten inhibitors the model
    was trained on.
    """
    chembl = np.asarray(chembl, dtype=float)
    drugs = np.asarray(drugs, dtype=float)
    similarity = np.array([[tan_sim(drug, compound) for drug in drugs]
                           for compound in chembl])
    return similarity.max(axis=1)
