# MD-DIF-hivreversetranscriptase

External validation code for the multi-drug (MD) drug–isolate fold change (DIF)
framework, applied to HIV-1 reverse transcriptase inhibitors.

> Yilmaz S, Tunc H, Sari M, Durdagi S. *Genotype–Compound Machine Learning for
> Multi-Drug Prediction of HIV-1 RTI Resistance.*

Models are trained on the Stanford HIV Drug Resistance Database and evaluated on
an independent ChEMBL-curated set of 805 isolate–compound observations. The
question the repository answers is how much of the predictive performance comes
from describing the *inhibitor* rather than the *isolate*, so three input
representations are compared on identical data, identical folds and identical
learners:

| Configuration | Input to the model | Purpose |
| --- | --- | --- |
| `Null-Model/` | 1388 isolate mutation bits | genotype-only reference |
| `Morgan-206/` | plus the 206 fingerprint bits that vary across the ten training RTIs | reduced inhibitor description |
| `Morgan-512/` | plus the full 512-bit Morgan fingerprint | complete inhibitor description |

Each is run with two learners — the multi-drug artificial neural network
(**ANN**) and gradient-boosted trees (**XGBoost**) — giving the six columns of
the external-validation table.

The null model receives no inhibitor identity, so a fixed trained model returns
the same value for every drug tested against one isolate:

> ŷ_null(I, d) = f_null(I) ≈ E_d[LFC(I, d) | I]

It therefore cannot rank two inhibitors on one genotype. That inability is
deliberate — it is what makes the null model a baseline against which the
contribution of the Morgan representation can be measured.

## Layout

Each configuration is one folder in two parts. `Training/` holds the inputs and
the model output; `Analysis/` holds the reference data and the scores. Every
folder carries its own copy of what its scripts open, so each stands alone and
can be run from its own directory with no package to install and nothing on the
import path.

```text
Null-Model/
    Training/     Final.csv.gz, folds.csv, YPRED_5_*.csv, YPRED_*.csv
        main.py          five-fold cross-validation -> YPRED_5_<learner>.csv
        training.py      one fold, either learner
        create_ypred.py  YPRED_5_<learner>.csv -> YPRED_<learner>.csv
    Analysis/     External_Data.csv, mutations.csv, morgan_drugs.csv,
                  morgan_chembl.csv.gz, morgan_map.csv, YPRED_*.csv
        analysis.py            scoring -> performance.csv
        class_perform.py       classification metrics
        str_char_improved.py   strain string -> mutation patterns
        tan_sim.py             Tanimoto similarity
Morgan-206/    identical
Morgan-512/    identical
```

The three configurations differ in exactly one thing: which columns of the
design matrix the model may see. `Training/main.py` states that in its
`columns()` function, and nothing else changes between them.

Keeping each folder self-contained means the shared modules are duplicated
across the three configurations. That is deliberate: a reader can take one
folder and run it, and there is no hidden state anywhere else in the tree.

## Running

```bash
pip install -r requirements.txt
```

Reproduce the published scores from the stored predictions — seconds, no
training, no GPU:

```bash
cd Morgan-512/Analysis && python analysis.py
```

`analysis.py` writes `performance.csv` beside itself and prints the same table.
Add `--algorithm ANN` to score one learner.

Retrain instead:

```bash
cd Morgan-512/Training
python main.py                      # both learners
python main.py --algorithm XGBoost  # one of them
python main.py --device cpu         # the network without a GPU
python create_ypred.py              # assemble the fold predictions
```

`main.py` refuses to replace the stored prediction files unless you pass
`--overwrite`, because those are what reproduce the published scores. On an
RTX 3080 the network takes roughly six minutes per configuration; XGBoost runs
on CPU.

`main.py` reads the archived fold split from `folds.csv`. `--refold` draws a
new one, which makes the run no longer comparable to the stored vectors.

## Data

Everything is CSV with named columns, so it can be read without any particular
toolchain and inspected in an editor. The two widest tables are gzipped, which
every CSV reader handles transparently. Row and column indices are zero-based.

`Training/`

| File | Contents |
| --- | --- |
| `Final.csv.gz` | 18841 rows. `Source` is Stanford or External, `Compound` names the inhibitor (a ChEMBL identifier for the external rows), `LFC` is the observed log10 fold change, then `mut_0001`–`mut_1388` for the isolate and `morgan_000`–`morgan_511` for the compound. Rows 0–18035 are Stanford (downloaded 22/02/2023), the remaining 805 are the external set. Each configuration reads the columns it is allowed to see. |
| `folds.csv` | `row, fold` — the five-fold split of the 805 external rows |
| `morgan_map.csv` | Morgan-206 only: the fingerprint bits that configuration may see |
| `YPRED_5_ANN.csv`, `YPRED_5_XGBoost.csv` | `fold, row, Predicted_log10FC` — predictions grouped by the fold that produced them |
| `YPRED_ANN.csv`, `YPRED_XGBoost.csv` | `row, Predicted_log10FC` — the same values in row order, one per external observation |

`Analysis/`

| File | Contents |
| --- | --- |
| `External_Data.csv` | the ChEMBL-curated external set, 1335 rows before filtering |
| `mutations.csv` | the 1388 unique mutations observed in the Stanford data |
| `morgan_drugs.csv` | 512-bit fingerprints of the 10 training inhibitors, one named row each |
| `morgan_chembl.csv.gz` | 512-bit fingerprints of the ChEMBL compounds |
| `morgan_map.csv` | the 206 fingerprint bits that vary across the training panel |
| `YPRED_ANN.csv`, `YPRED_XGBoost.csv` | a copy of the prediction vectors, so the folder scores without reaching into `Training/` |

Float columns are written with seventeen significant digits and read back with
`float_precision="round_trip"`. This is not fussiness: eight observations sit
exactly on the resistance threshold, and a parser that drops the last digit
moves them to the other side of it and shifts every classification metric.

## Method

**Training.** The 805 external observations are split into five folds. Each fold
in turn is held out and the model is trained on all 18036 Stanford rows plus the
four remaining external folds, then predicts the held-out fold.

The network is a single hidden layer of five tanh units, inputs and targets
scaled to [-1, 1], trained with mini-batch Adam and early stopping on a held-out
ten per cent. For each of ten ensemble members, five networks are trained from
independent random initialisations and the one with the lowest error on its
internal test split is kept; the ten retained networks are averaged. All fifty
candidates of a fold are trained together as one batched tensor, so a fold is a
single GPU job rather than fifty.

XGBoost uses 2500 trees, learning rate 0.05, maximum depth 8, minimum child
weight 1, histogram tree construction. No row or column subsampling is enabled,
so it is deterministic given the data and the seed.

**Scoring.** Predictions and observations are compared in log10 fold-change
units. An observation is resistant at or above a three-fold change, and the same
threshold turns a predicted value into a predicted class, so the regression
model is scored as a classifier without refitting anything.

The comparison is `>=`, which gives 463 resistant and 342 susceptible
observations. Eight observations sit exactly on the threshold, so scoring them
with a strict `>` instead would move all eight to susceptible, give 455 / 350,
and change most cells of the table below. The `>=` labelling is the one the
published metrics were computed with, so it is what the scripts use, and it
reproduces the published table exactly.

**Row selection.** `Final.csv.gz` stores the external block already encoded.
`analysis.py` rebuilds which spreadsheet rows those 805 observations are, from
`External_Data.csv` alone, and checks the result against `Final.csv.gz` when
that file is present — response, isolate encoding and fingerprint all have to
match. Two filters run, in this order:

1. *Assay filter.* Activity and wild type must both be present and the activity
   must be non-zero. Where the `FoldChange` column reads `No` the response is
   activity over wild type; otherwise the activity is already a fold change.
2. *Encoding filter.* Each strain string is expanded into individual mutation
   patterns and matched against the mutation list; a row is dropped unless the
   match count equals the number of patterns. Since that list holds each
   mutation once, the count never exceeds one, so the filter keeps only isolates
   carrying a single mutation pattern.

1335 rows reduce to 805. The second filter is stricter than it may look — it is
what defines the external set as single-mutation isolates.

## Reproduction

`cd <configuration>/Analysis && python analysis.py` returns:

| Metric | ANN Null | ANN M-512 | ANN M-206 | XGB Null | XGB M-512 | XGB M-206 |
| --- | --- | --- | --- | --- | --- | --- |
| Accuracy | 0.646 | **0.784** | 0.774 | 0.632 | **0.824** | 0.816 |
| Sensitivity | 0.877 | **0.829** | 0.810 | 0.819 | **0.849** | 0.829 |
| Specificity | 0.333 | 0.722 | **0.725** | 0.380 | 0.789 | **0.798** |
| Precision | 0.640 | **0.802** | 0.800 | 0.641 | 0.845 | **0.848** |
| Recall | 0.877 | **0.829** | 0.810 | 0.819 | **0.849** | 0.829 |
| F1-score | 0.740 | **0.815** | 0.805 | 0.719 | **0.847** | 0.838 |
| AUC | 0.678 | **0.870** | 0.860 | 0.668 | **0.901** | 0.896 |
| r | 0.455 | **0.714** | 0.708 | 0.468 | 0.773 | **0.777** |
| MCC | 0.254 | **0.555** | 0.536 | 0.222 | **0.639** | 0.625 |
| AUPRC | 0.735 | **0.906** | 0.902 | 0.729 | **0.925** | 0.923 |

Bold marks the better of the two fingerprint configurations. The null model is
excluded from that comparison: it takes the highest sensitivity merely by
calling almost everything resistant, and its specificity of 0.333 and 0.380
shows that.

What is guaranteed, and what is not:

| | Reproducible? |
| --- | --- |
| Scoring the stored predictions | exactly, on any machine, in seconds |
| Retraining **XGBoost** | exactly — no subsampling is enabled, so it is a deterministic function of the data and the seed |
| Retraining the **ANN** | to within run-to-run variation. Initialisation is random and is not seeded to a fixed draw, so the numbers land near the published ones rather than on them |

The XGBoost guarantee holds within a major version; boosting internals change
between releases. The versions the published results were produced with are
listed at the end of `requirements.txt`.

## Requirements

Python 3.10 or newer and the packages in `requirements.txt`. PyTorch is needed
only to retrain the network; scoring the stored predictions does not use it.
