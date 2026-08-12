"""Cross-validate the external set - Morgan-206.

The reduced drug-isolate model: the model sees the 1388-bit isolate mutation
vector and the 206 fingerprint bits that vary across the ten training
inhibitors. The other 306 carry no information the training panel can learn
from, so dropping them costs nothing on this panel - but it also means a new
compound is described only through features the panel already varies in, which
is the transferability caveat the manuscript raises.

The 805 external observations are split into five folds. Each fold in turn is
held out and the model is trained on all 18036 Stanford rows plus the four
remaining external folds, then predicts the held-out fold.

    python main.py                      both learners
    python main.py --algorithm XGBoost  one of them
    python main.py --device cpu         the network without a GPU

Writes the per-fold predictions to ``YPRED_5_<learner>.csv``; ``create_ypred.py``
then scatters those back to row order. Existing prediction files are kept
unless ``--overwrite`` is given, because they are what reproduces the published
scores.

The fold split is read from ``folds.csv`` - the archived split the published
predictions were produced with. ``--refold`` draws a new one instead, which
makes the run no longer comparable to the stored vectors.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from training import training

HERE = Path(__file__).resolve().parent
REPRESENTATION = "Morgan-206"

ALGORITHMS = ("ANN", "XGBoost")

#: rows 0..18035 of ``Final.csv.gz`` are Stanford; the remaining 805 are ChEMBL
N_STANFORD = 18_036
N_EXTERNAL = 805

#: column 0 is the target; then 1388 mutation bits, then 512 fingerprint bits
N_MUTATIONS = 1_388
N_MORGAN = 512

#: of those 512, the ones that vary across the ten training inhibitors
N_MORGAN_206 = 206

N_FOLDS = 5

#: enough digits for a float64 to survive the round trip through text
FLOAT_FORMAT = "%.17g"


def read_csv(path, **kwargs) -> pd.DataFrame:
    """``pd.read_csv`` that recovers float64 values exactly."""
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def load_final() -> np.ndarray:
    """The Stanford + ChEMBL design matrix, 18841 x 1901, target in column 0."""
    frame = read_csv(HERE / "Final.csv.gz").drop(columns=["Source", "Compound"])
    final = frame.to_numpy(dtype=np.float64)
    expected = (N_STANFORD + N_EXTERNAL, 1 + N_MUTATIONS + N_MORGAN)
    if final.shape != expected:
        raise RuntimeError(f"Final.csv.gz is {final.shape}, expected {expected}")
    if frame.columns[0] != "LFC":
        raise RuntimeError("Final.csv.gz does not start with the LFC column")
    return final


def columns() -> np.ndarray:
    """Zero-based column indices this configuration is allowed to see.

    The isolate bits, plus the fingerprint positions listed in
    ``morgan_map.csv`` - the 206 bits that are not constant across the ten
    training inhibitors. That single restriction is the whole difference from
    Morgan-512.
    """
    keep = read_csv(HERE / "morgan_map.csv")["bit"].to_numpy(np.int64)
    if keep.size != N_MORGAN_206:
        raise RuntimeError(f"morgan_map.csv selects {keep.size} bits, "
                           f"expected {N_MORGAN_206}")
    return np.concatenate((np.arange(N_MUTATIONS), N_MUTATIONS + keep))


def load_folds() -> tuple[np.ndarray, ...]:
    """The archived five-fold split of the external rows, zero-based."""
    frame = read_csv(HERE / "folds.csv")
    folds = tuple(np.sort(frame.loc[frame["fold"] == k, "row"].to_numpy(np.int64))
                  for k in sorted(frame["fold"].unique()))
    if not np.array_equal(np.sort(np.concatenate(folds)), np.arange(N_EXTERNAL)):
        raise RuntimeError("folds.csv does not cover the "
                           f"{N_EXTERNAL} external rows")
    return folds


def draw_folds(seed: int = 0) -> tuple[np.ndarray, ...]:
    """A fresh five-fold split of the external rows, balanced in size."""
    rng = np.random.default_rng(seed)
    assignment = rng.permutation(np.arange(N_EXTERNAL) % N_FOLDS)
    folds = tuple(np.flatnonzero(assignment == k) for k in range(N_FOLDS))
    pd.DataFrame({"row": np.arange(N_EXTERNAL),
                  "fold": assignment + 1}).to_csv(HERE / "folds.csv", index=False)
    print(f"wrote a new split to folds.csv (seed {seed})")
    return folds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=f"Cross-validate the external set ({REPRESENTATION}).")
    parser.add_argument("--algorithm", nargs="*", choices=ALGORITHMS,
                        help="default: both")
    parser.add_argument("--device", default="cuda", help="the network only")
    parser.add_argument("--n-jobs", type=int, default=4, help="XGBoost only")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace prediction files that already exist")
    parser.add_argument("--refold", action="store_true",
                        help="draw a new fold split instead of reading folds.csv")
    args = parser.parse_args(argv)

    wanted = list(args.algorithm) if args.algorithm else list(ALGORITHMS)

    # the stored vectors are what reproduces the published scores, so they are
    # not replaced by accident; retraining the network will not return them exactly
    existing = [a for a in wanted
                if (HERE / f"YPRED_{a}.csv").exists()
                or (HERE / f"YPRED_5_{a}.csv").exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"predictions for {', '.join(existing)} already exist in {HERE}.\n"
            f"These are the published vectors that reproduce the manuscript "
            f"scores. Pass --overwrite to replace them.")

    final = load_final()
    folds = draw_folds() if args.refold else load_folds()
    cols = columns()

    features, y = final[:, 1:], final[:, 0]
    x_stanford, y_stanford = features[:N_STANFORD], y[:N_STANFORD]
    x_external, y_external = features[N_STANFORD:], y[N_STANFORD:]

    for algorithm in wanted:
        records = []
        for f, test in enumerate(folds):
            train = np.setdiff1d(np.arange(N_EXTERNAL), test, assume_unique=True)
            x_tr = np.ascontiguousarray(
                np.vstack((x_stanford[:, cols], x_external[train][:, cols])))
            y_tr = np.concatenate((y_stanford, y_external[train]))
            x_te = np.ascontiguousarray(x_external[test][:, cols])

            started = time.time()
            predicted = training(x_tr, y_tr, x_te, algorithm, f,
                                 device=args.device, n_jobs=args.n_jobs)
            print(f"{REPRESENTATION:11} {algorithm:8} fold {f + 1}/{len(folds)}  "
                  f"train {len(y_tr):5d}  test {len(test):3d}  "
                  f"{time.time() - started:6.1f}s", flush=True)
            records.append(pd.DataFrame({"fold": f + 1, "row": test,
                                         "Predicted_log10FC": predicted}))

        path = HERE / f"YPRED_5_{algorithm}.csv"
        pd.concat(records).to_csv(path, index=False, float_format=FLOAT_FORMAT)
        print(f"wrote {path.name}", flush=True)

    print("now run:  python create_ypred.py")


if __name__ == "__main__":
    main()
