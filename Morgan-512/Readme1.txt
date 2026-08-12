Morgan-512: the full drug-isolate model. The model sees the 1388-bit isolate
mutation vector and the complete 512-bit Morgan fingerprint of the inhibitor, so
one model covers all ten reverse transcriptase inhibitors and can tell two drugs
apart on the same genotype.

Training folder: main.py cross-validates the external set using Final.csv.gz,
which holds the Stanford HIV Drug Resistance and ChEMBL data. Each fold's
predictions are written to YPRED_5_<learner>.csv, and create_ypred.py assembles
them into YPRED_<learner>.csv, the key output of this folder.
Analysis folder: model performance is evaluated with analysis.py using the
YPRED_<learner>.csv files, and written to performance.csv.

# Data:
External_Data.csv: ChEMBL-curated dataset. Refer to the manuscript for full details.
Final.csv.gz: Includes Stanford data for 10 RTIs (downloaded 22/02/2023) and the ChEMBL-curated external dataset, already encoded. Column LFC is the observed log10 fold change; mut_0001-mut_1388 describe the isolate and morgan_000-morgan_511 the compound.
mutations.csv: Contains the 1388 unique mutations found in the Stanford dataset.
YPRED_5_ANN.csv, YPRED_5_XGBoost.csv: Training results obtained using 5-fold cv, grouped by fold (result of main.py).
folds.csv: Contains the indices of the 5-fold cv.
YPRED_ANN.csv, YPRED_XGBoost.csv: Out-of-fold estimates in row order, obtained as a result of cross-validation (result of create_ypred.py). A copy is kept in Analysis so that folder stands alone.
morgan_drugs.csv: Representation of inhibitors for the Stanford 10 RTIs.
morgan_chembl.csv.gz: Representation of inhibitors for the ChEMBL database.
morgan_map.csv: The 206 fingerprint bits that vary across the training panel. Used by the Morgan-206 configuration; this one reads all 512.

# Codes:
str_char_improved.py: This function converts the isolates into individual mutation patterns.
class_perform.py: This file calculates classification metrics.
tan_sim.py: This file measures the Tanimoto similarity of two fingerprint vectors.
training.py: This file conducts the training of one fold, with the ANN or with XGBoost.
main.py: This code generates predictions by cross-validating the external dataset and saves these predictions to a file. Its columns() function is what selects the representation.
create_ypred.py: YPRED_5_<learner>.csv and folds.csv are loaded. It rearranges the fold predictions into row order and saves them to YPRED_<learner>.csv.
analysis.py: This code contains the set of operations used to analyse the results, and writes performance.csv.
