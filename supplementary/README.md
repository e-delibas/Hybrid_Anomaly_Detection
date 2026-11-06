# Supplementary Data S1. Sensitivity to the RUL reference window

**Manuscript:** Evaluating Autoencoder Representations for Anomaly Detection: An Explainable Hybrid Framework

**Manuscript ID:** JER-D-26-01469

This dataset consolidates the primary RUL-30 evaluation and the RUL-20 and RUL-40 sensitivity evaluations reported in Sections 2.3, 4.2, 4.9 and 4.10. It contains existing results; no models were refitted for this supplement.

## Evaluation reference and interpretation

For each engine, RUL is its final recorded cycle minus the current cycle. A cycle is positive when 0 ≤ RUL ≤ w, where w is 20, 30 or 40 cycles. With consecutive cycles, the inclusive definition selects w + 1 terminal observations per engine. These labels indicate proximity to the recorded end of life. They are not independently annotated fault-onset labels, and a negative reference label does not establish that a cycle is physically healthy.

The primary window is 30 cycles. The 20- and 40-cycle evaluations retain the models, scores, selected detector parameters and decision thresholds from the primary protocol. Only the evaluation reference changes. These results measure sensitivity to the reference window, not the performance of models separately optimized for each window.

Results are equal-weight means and descriptive sample standard deviations (ddof = 1) over ten engine-disjoint outer folds and three model seeds (42, 43 and 44). Each aggregate row has 30 contributing fold/seed records. The folds are shared across seeds, and fitting sets overlap across outer folds. These 30 records are not independent datasets. No pooled-score AP or significance test is implied.

## Files

| File | Contents |
|---|---|
| Supplementary_Data_S1_RUL_Sensitivity.xlsx | Filterable aggregate metrics and protocol notes. |
| AP_F1_by_RUL_window.csv | 154 representation/method/settings combinations, with AP and F1 means and descriptive SDs for all three windows side by side. |
| aggregate_metrics.csv | 462 aggregate rows, including all six metrics and their descriptive SDs. |
| fold_metrics.csv | 13,860 fold/seed/window records, confusion counts, thresholds and selected parameters. These are fold summaries, not individual-cycle predictions. |
| source_provenance.json | Source-summary filenames, SHA-256 fingerprints, experiment-plan identifiers and export-validation scope. |
| verify_supplement.py | Standard-library Python script that checks the CSV exports and recomputes aggregate means and descriptive SDs from the fold records. |
| SHA256SUMS.txt | SHA-256 fingerprints for the other package files. |

## Experimental coverage

| Source stage | Dataset and preprocessing | Coverage | Aggregate rows | Fold records |
|---|---|---|---:|---:|
| 07C | FD001, global standardization | 12 representations × 5 classical detectors, plus reconstruction-error controls for 6 AEs; 3 windows | 198 | 5,940 |
| 10C | FD002 global; FD002 conditionwise; FD003 global | In each setting: 5 representations × 5 classical detectors, plus reconstruction-error controls for 2 AEs; 3 windows | 243 | 7,290 |
| 15C | FD001, global standardization | SCLM-64 × 5 classical detectors and 2 complete SEAD variants; 3 windows | 21 | 630 |
| Total | | | 462 | 13,860 |

The five classical detectors are DBSCAN, IF, LOF, OC-SVM and kNN. MSE is a separate reconstruction-error control, not a sixth classical detector. FD002 conditionwise preprocessing is a separate experimental setting, not an additional dataset. No FD003 conditionwise experiment is included.

## Labels and columns

- `Raw`: standardized 21-sensor input.
- `PCA_d4`, `PCA_d8`, `PCA_d21`: PCA with 4, 8 or 21 components, without component-wise output standardization or whitening.
- `PCA_scaled_d4`, `PCA_scaled_d8`: PCA followed by component-wise standardization fitted on fitting-engine representations.
- `AE_deep_d4/d8/d21`, `AE_shallow_d4/d8/d21`: deep or shallow AE latent representations with the stated dimension.
- `SCLM64`: the pretrained 64-dimensional self-contrastive representation supplied to the five classical detectors.
- `SEAD_full`: the complete weakly supervised scoring model. `SEAD_adapted_raw_error` is the primary adaptation; `SEAD_adapted_unit_direction_control` retains the source implementation's normalized-residual reconstruction penalty. Both use the additional fitting-engine positive proxy labels described in Section 4.10. Architecture, representation size, scoring, supervision and training budgets differ from the Raw/PCA/AE comparisons.
- `method = DBSCAN`: the continuous fixed-core distance score described in the manuscript, evaluated using a selected alarm threshold.
- `AP`: average precision; `ROC_AUC`: area under the ROC curve; `precision`, `recall` and `F1`: positive-reference-class metrics; `Macro_F1`: arithmetic mean of the two class-specific F1 values.
- `*_descriptive_sd`: sample standard deviation of the 30 fold/seed values, not a standard error or confidence interval.
- `TP`, `FP`, `TN`, `FN`: counts relative to the RUL reference for that row.
- `threshold`: the original detector-specific decision threshold, retained across all three RUL windows. Threshold magnitudes are not comparable across different detector score scales.
- `selected_parameters_json`: the recorded detector configuration serialized as JSON; an empty object means that no separate detector configuration was recorded for that score in the source fold summary.
- `source_stage`: the archived experiment-summary stage. Each aggregate row and fold row has one source stage; baseline results already carried forward into 07C are not counted again from earlier stages.

## Verification and scope

Run `python verify_supplement.py` in the extracted package directory. The script requires only the Python standard library and does not train models or alter the package.

The export was checked for unique keys, complete ten-fold/three-seed coverage, preservation of all five classical detectors, consistency of confusion-derived metrics, agreement of aggregate means and sample SDs with the fold records, and unchanged thresholds, selected parameters, sample counts and alarm counts across reference windows. AP and ROC-AUC are retained from the archived experiment results; this supplement does not recompute them from individual-cycle scores. Equal alarm counts alone do not establish identity of individual predictions. The experiment code and archived run protocol govern the use of unchanged models and scores.

The workbook displays numerical values to six decimal places; CSV files retain the exported numerical precision. The original three source-summary JSON files are identified by exact fingerprints in `source_provenance.json`. Their full execution logs and local filesystem paths are not included here. This supplement is a numerical results package, not the full training-code or model-artifact release.
