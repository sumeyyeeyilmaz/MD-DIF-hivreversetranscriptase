"""Score the out-of-fold predictions - Morgan-206.

The reduced drug-isolate model: the model sees the 1388-bit isolate mutation
vector and the 206 fingerprint bits that vary across the ten training
inhibitors. The other 306 carry no information the training panel can learn
from, so dropping them costs nothing on this panel - but it also means a new
compound is described only through features the panel already varies in, which
is the transferability caveat the manuscript raises.

    python analysis.py
    python analysis.py --algorithm XGBoost

Reads the prediction vectors stored beside this script and writes
``performance.csv``. Nothing is retrained, so the published scores come back in
seconds and without a GPU.

This folder stands alone - it opens nothing outside itself, except that when
``../Training/Final.csv.gz`` happens to be present the reconstructed row
selection is checked against it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)

from class_perform import class_perform
from str_char_improved import str_char_improved
from tan_sim import nearest_training_similarity

HERE = Path(__file__).resolve().parent
REPRESENTATION = "Morgan-206"

ALGORITHMS = ("ANN", "XGBoost")

#: an isolate-compound observation is resistant above a three-fold change
RESISTANCE_FC = 3.0
THRESHOLD = float(np.log10(RESISTANCE_FC))

#: the order the manuscript prints them in
METRICS = ("Accuracy", "Sensitivity", "Specificity", "Precision", "Recall",
           "F1-score", "AUC", "r", "MCC", "AUPRC")

#: ``External_Data.csv`` headers, mapped to identifier-safe names
COLUMNS = {
    "Drug/Compound": "Drug_Compound",
    "Wild Type": "WildType",
    "Assay Type": "AssayType",
    "Fold Change": "FoldChange",
}

N_EXTERNAL = 805
N_MUTATIONS = 1_388
N_STANFORD = 18_036


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """``pd.read_csv`` that recovers float64 values exactly.

    The default parser is fast but drops the last digit, which is not harmless
    here: eight observations sit exactly on the resistance threshold, and
    losing a digit moves them to the other side of it and shifts every
    classification metric.
    """
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def select_external():
    """The external observations, filtered as the published analysis filters.

    Two filters run, in this order.

    First, an assay filter: activity and wild type must both be present and the
    activity must be non-zero. Where the ``FoldChange`` column says ``No`` the
    response is activity over wild type, otherwise the activity is already a
    fold change and is taken as it stands.

    Second, an encoding filter: each strain string is expanded into individual
    mutation patterns and matched against the 1388 Stanford mutations, and a
    row is dropped unless the match count equals the number of patterns.
    Because that list holds each mutation once the count never exceeds one, so
    the filter keeps only isolates carrying a single mutation pattern. This is
    the behaviour that produced the published 805-row external set, and it is
    reproduced here rather than corrected.

    1335 rows reduce to 805.
    """
    table = read_csv(HERE / "External_Data.csv").rename(columns=COLUMNS)
    mutations = read_csv(HERE / "mutations.csv")["mutation"].tolist()

    activity = pd.to_numeric(table["Activity"], errors="coerce").to_numpy(float)
    wild_type = pd.to_numeric(table["WildType"], errors="coerce").to_numpy(float)
    fold_change = table["FoldChange"].astype(str).to_numpy()
    strain = table["Strain"].astype(str).to_numpy()

    usable = np.flatnonzero(~np.isnan(activity) & ~np.isnan(wild_type)
                            & (activity != 0))
    response = np.where(fold_change[usable] == "No",
                        activity[usable] / wild_type[usable], activity[usable])

    index = {name: j for j, name in enumerate(mutations)}
    encoded = np.zeros((usable.size, len(mutations)), dtype=np.float64)
    keep = np.zeros(usable.size, dtype=bool)
    for i, text in enumerate(strain[usable]):
        patterns = str_char_improved(text)
        matched = 0
        for pattern in patterns:
            column = index.get(pattern)
            matched = 0 if column is None else 1
            if column is not None:
                encoded[i, column] = 1.0
        keep[i] = matched == len(patterns)

    return usable[keep], response[keep], encoded[keep]


def evaluate(ypred: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    """Every published metric for one out-of-fold prediction vector.

    Predictions and observations are compared in log10 fold-change units, and
    the same threshold that calls an observation resistant turns a predicted
    value into a predicted class - so the regression model is scored as a
    classifier without refitting anything.
    """
    ypred = np.asarray(ypred, dtype=np.float64).ravel()
    observed = np.asarray(observed, dtype=np.float64).ravel()
    if ypred.size != observed.size:
        raise ValueError(f"{ypred.size} predictions for {observed.size} rows")

    truth = (observed >= THRESHOLD).astype(int)
    hard = (ypred >= THRESHOLD).astype(int)
    counts = class_perform(hard, truth)
    tn, fp, fn, tp = confusion_matrix(truth, hard, labels=(0, 1)).ravel()

    scores = counts.as_dict()
    scores.update({
        "AUC": float(roc_auc_score(truth, ypred)),
        "r": float(np.corrcoef(observed, ypred)[0, 1]),
        "MCC": float(matthews_corrcoef(truth, hard)),
        "AUPRC": float(average_precision_score(truth, ypred)),
    })
    return scores


def _verify(rows: np.ndarray, response: np.ndarray,
            encoded: np.ndarray) -> None:
    """Check the reconstruction against ``Final.csv.gz`` when it is there."""
    path = HERE.parent / "Training" / "Final.csv.gz"
    if not path.exists():
        return

    frame = read_csv(path).drop(columns=["Source", "Compound"])
    final = frame.to_numpy(dtype=np.float64)[N_STANFORD:]
    if not np.allclose(final[:, 0], np.log10(response), rtol=0, atol=1e-9):
        raise RuntimeError("the reconstructed response does not match Final.csv.gz")
    if not np.array_equal(final[:, 1:1 + N_MUTATIONS], encoded):
        raise RuntimeError("the reconstructed isolates do not match Final.csv.gz")

    chembl = read_csv(HERE / "morgan_chembl.csv.gz").drop(
        columns="row").to_numpy(np.float64)
    if not np.array_equal(final[:, 1 + N_MUTATIONS:], chembl[rows]):
        raise RuntimeError("the reconstructed compounds do not match Final.csv.gz")
    print("row selection verified against Final.csv.gz")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=f"Score the external predictions ({REPRESENTATION}).")
    parser.add_argument("--algorithm", nargs="*", choices=ALGORITHMS,
                        help="default: both")
    args = parser.parse_args(argv)

    rows, response, encoded = select_external()
    if rows.size != N_EXTERNAL:
        raise RuntimeError(f"selected {rows.size} rows, expected {N_EXTERNAL}")
    observed = np.log10(response)
    truth = (observed >= THRESHOLD).astype(int)
    print(f"{REPRESENTATION}: {rows.size} external observations, "
          f"{int(truth.sum())} resistant, {int((1 - truth).sum())} susceptible")
    _verify(rows, response, encoded)

    # how far the external compounds sit from the ten training inhibitors
    drugs = read_csv(HERE / "morgan_drugs.csv")
    chembl = read_csv(HERE / "morgan_chembl.csv.gz").drop(columns="row")
    similarity = nearest_training_similarity(
        chembl.to_numpy(np.float64)[rows], drugs.drop(columns="drug").to_numpy(np.float64))
    print(f"Tanimoto similarity to the nearest of the {len(drugs)} training "
          f"RTIs: median {np.median(similarity):.3f}, "
          f"range {similarity.min():.3f}-{similarity.max():.3f}")

    scores = {}
    for algorithm in (args.algorithm or list(ALGORITHMS)):
        path = HERE / f"YPRED_{algorithm}.csv"
        if not path.exists():
            path = HERE.parent / "Training" / f"YPRED_{algorithm}.csv"
        if not path.exists():
            print(f"  {algorithm}: YPRED_{algorithm}.csv not found, "
                  f"run Training first")
            continue
        frame = read_csv(path).sort_values("row")
        ypred = frame["Predicted_log10FC"].to_numpy(np.float64)
        if ypred.size != N_EXTERNAL:
            raise RuntimeError(f"{path} holds {ypred.size} predictions, "
                               f"expected {N_EXTERNAL}")
        scores[algorithm] = evaluate(ypred, observed)

    if not scores:
        return
    table = pd.DataFrame(scores).reindex(METRICS)
    table.index.name = "Metric"
    table.round(3).to_csv(HERE / "performance.csv")
    print(table.round(3).to_string())
    print(f"wrote performance.csv to {HERE}")


if __name__ == "__main__":
    main()
