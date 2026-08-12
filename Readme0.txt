Our research adopts the null model approach for the proposed DIF models, which evaluates the impact of drug information on the model's prediction performance.
The null model serves as a reference point to evaluate the accuracy and reliability of the model by excluding certain drug information. It provides the performance
metrics reported in the Manuscript.

Three configurations are compared on identical data, identical folds and identical learners, and each is run with both the ANN and XGBoost:
Null-Model: the isolate mutation vector only, no inhibitor information.
Morgan-206: plus the 206 fingerprint bits that vary across the ten training RTIs.
Morgan-512: plus the full 512-bit Morgan fingerprint.

Stanford_Data: Stanford data for 10 RTIs (downloaded 22/02/2023). Distributed here already encoded, as the numeric block of Training/Final.csv.gz.
External_Data.csv: ChEMBL-curated dataset. Refer to the manuscript for full details.

The code is Python and every data file is CSV, so a folder can be run as it
stands. See README.md for the layout and for how the published scores are
reproduced from the stored prediction vectors.
