"""Assemble the per-fold predictions into one out-of-fold vector.

``main.py`` writes each fold's predictions against the rows that fold held out;
this scatters them back into row order, giving one prediction per external
observation, and drops a copy next to ``analysis.py`` so the Analysis folder
stands alone.

    python create_ypred.py

``YPRED_5_<learner>.csv`` -> ``YPRED_<learner>.csv`` and
``../Analysis/YPRED_<learner>.csv``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

ALGORITHMS = ("ANN", "XGBoost")

N_EXTERNAL = 805

#: enough digits for a float64 to survive the round trip through text
FLOAT_FORMAT = "%.17g"


def assemble(algorithm: str) -> np.ndarray:
    """Scatter one learner's fold predictions back to row order."""
    frame = pd.read_csv(HERE / f"YPRED_5_{algorithm}.csv",
                        float_precision="round_trip")

    ypred = np.full(N_EXTERNAL, np.nan)
    ypred[frame["row"].to_numpy(np.int64)] = frame[
        "Predicted_log10FC"].to_numpy(np.float64)
    missing = int(np.isnan(ypred).sum())
    if missing:
        raise RuntimeError(f"{missing} external rows have no prediction")
    return ypred


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Assemble per-fold predictions into one vector.")
    parser.add_argument("--algorithm", nargs="*", choices=ALGORITHMS,
                        help="default: both")
    args = parser.parse_args(argv)

    analysis = HERE.parent / "Analysis"
    for algorithm in (args.algorithm or list(ALGORITHMS)):
        if not (HERE / f"YPRED_5_{algorithm}.csv").exists():
            print(f"  {algorithm}: YPRED_5_{algorithm}.csv not found, "
                  f"run main.py first")
            continue

        ypred = assemble(algorithm)
        path = HERE / f"YPRED_{algorithm}.csv"
        pd.DataFrame({"row": np.arange(ypred.size),
                      "Predicted_log10FC": ypred}).to_csv(
            path, index=False, float_format=FLOAT_FORMAT)
        shutil.copyfile(path, analysis / path.name)
        print(f"wrote {path.name} and ../Analysis/{path.name}")


if __name__ == "__main__":
    main()
