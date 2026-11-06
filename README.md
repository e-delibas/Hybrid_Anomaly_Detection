# Autoencoder representations for anomaly detection

Code for **Evaluating Autoencoder Representations for Anomaly Detection: An Explainable Hybrid Framework**.

This release candidate reorganizes the authors' 28 executed revision notebooks. It retains the recorded experiment plans, detector definitions and source checks. Large plans and worker programs are stored under `assets/`; notebook outputs have been cleared. `notebook_map.csv` maps each release notebook to its original filename.

## Current scope

The notebooks document and replay the recorded research workflow. Several completion stages require previously saved representations, models and pilot results with the exact fingerprints recorded in their plans. They are **not independent fresh-run notebooks**. Running every notebook in filename order on a new computer is not a supported reproduction procedure.

The package currently supports:

- inspecting the full experiment code and fixed plans;
- checking the supplied aggregate/fold results without model training;
- checking whether the local archived experiment files match the completion-stage dependencies;
- running recorded stages when their prerequisites and environments are available.

The reorganized notebooks have passed structural source comparisons, not a complete model-training replay. A clean-install fresh-training workflow remains to be validated. Archived weights and per-cycle prediction files are not included in this package.

## Initial check

Place this folder inside the existing project that contains `data/` and `revision_outputs/`. Open `00_Check_Recorded_Run.ipynb` and run its code cell. Alternatively:

```bash
python tools/preflight.py --project /path/to/experiment/project
```

The check reads and hashes the listed files, checks summary fingerprints and inspects installed package versions. It does not import TensorFlow or PyTorch, fit models, load serialized estimators or modify existing experiment results. It writes a new report below `revision_outputs/16_code_release_preflight/`.

`RECORDED_DEPENDENCIES_VERIFIED` means the listed source files, summary anchors, release files and recorded package versions matched. It does not establish fresh-training reproduction or successful model imports. `REVIEW_REQUIRED` identifies missing or changed files or unmatched environments.

## Paths and environments

Copy `settings.example.json` to `settings.json` when automatic project discovery is insufficient. Set `project_dir`, `classical_python` and `torch_python`. Environment variables `CMAPSS_PROJECT_DIR`, `CMAPSS_CLASSICAL_PYTHON` and `CMAPSS_TORCH_PYTHON` override these settings. Notebook kernels otherwise supply the default Python executable; the recorded stages retain their own compatibility checks.

The recorded environments differ:

| Environment | Python | Main packages |
|---|---|---|
| Classical detectors and TensorFlow AE | 3.12.16 | NumPy 2.1.3; pandas 2.2.3; scikit-learn 1.6.1; TensorFlow 2.21.0; Keras 3.14.0; SciPy 1.18.1; joblib 1.6.0 |
| SEAD neural training | 3.13.5 | PyTorch 2.11.0; NumPy 2.1.3; pandas 2.2.3; scikit-learn 1.6.1; SciPy 1.15.3; joblib 1.4.2 |

`environments/*_recorded.txt` records these core versions. These files are not full transitive environment locks and do not claim a verified installation on every platform. Preserve the two environments for recorded-run checks. Plotting dependencies also appear in the figure notebooks; they are not covered by these core version lists.

## Experiment stages

| Stages | Role |
|---|---|
| 01, 02 | FD001 input validation and engine-disjoint reference protocol |
| 03, 04, 05 | Initial FD001 pilot and baseline representation production |
| 02B, 02C | Integrity and score checks for the archived representations |
| 03B, 05B | Five-detector pilot and complete FD001 Raw/AE-21 comparison |
| 06, 06B, 06C | AE dimension/architecture production and five-detector completion |
| 07, 07B, 07C | PCA production and five-detector completion |
| 08, 09 | FD002/FD003 input validation and engine protocols |
| 10, 10B1, 10C | Additional representations and five-detector completion |
| 10B | Historical pilot before the resource amendment; retained for provenance |
| 11, 12, 13, 14 | ROC, fixed/selected parameters, t-SNE and detector SHAP |
| 15A, 15B, 15C | SEAD preflight, training/selection pilot and full CV |

Earlier representation-producing stages contain the initial three-detector work. Their outputs are dependencies of the later five-detector completion stages, not the final comparison alone. The final scope retains **DBSCAN, IF, LOF, OC-SVM and kNN**. AE reconstruction error remains a separate control.

| Manuscript output | Final experiment source |
|---|---|
| Tables 6–7; Figure 4 | 05B FD001 baseline |
| Figure 5; representation comparisons | 06C and 07C |
| Figure 6 | 11, using recorded baseline predictions |
| Tables 10–11 | 12 fixed/selected parameter comparison |
| Figure 7 | 13 t-SNE |
| Figure 8 | 14 detector SHAP |
| Table 13 | 10C additional subsets |
| Table 14 | 15C SCLM/full-SEAD comparison |
| Supplementary Data S1 | 07C, 10C and 15C fold summaries |

## Scientific conventions

Engine identities define the fitting, early-stopping, selection and outer-evaluation partitions. Preprocessing and representation fitting use fitting engines only. The primary reference is inclusive RUL ≤ 30; RUL ≤ 20 and ≤ 40 change the evaluation reference while retaining fitted models, scores and selected thresholds. RUL labels indicate proximity to the recorded end of life, not annotated fault onset.

The complete SEAD adaptation uses additional positive fitting-engine proxy labels. Its supervision, representation size, architecture and scoring differ from the classical-detector comparisons. Both the primary and source-loss-control branches are retained.

## Supplementary data

The manuscript's **Supplementary Data S1 is the Excel workbook** `Supplementary_Data_S1_RUL_Sensitivity.xlsx`. The journal submission can use that workbook alone. `supplementary/` additionally provides the CSV fold/aggregate exports, source fingerprints and the standard-library verification script for code-repository users:

```bash
python supplementary/verify_supplement.py
```

These checks recompute aggregate means and descriptive sample SDs from the recorded fold values. AP and ROC-AUC are not recalculated from individual-cycle scores by this script.

## Source correspondence

The numerical values in externally stored plans and the text of extracted worker programs are unchanged. `source_audit.json` records correspondence to the uploaded notebooks. After reversing the documented path substitutions and external literal loaders, the generated code's AST matches the original code. Short English notebook introductions replace revision-conversation text. Historical diagnostic/error strings and hash-bound provenance remain unchanged.

The source correspondence check does not replace runtime validation. Never disable hash checks to make an incompatible archived run pass. Preserve the original experiment outputs separately from new experiments.

## SEAD attribution

The SEAD comparison adapts *Self-Contrastive Learning to Boost Weakly Supervised Anomaly Detection*, DOI `10.1109/TNSM.2026.3694683`, and the released implementation at <https://github.com/pumpkinyouge/SEAD-Net>, commit `edd3162d949fe1e005721c7c400c5f52852ab6ae`. The adaptation and primary/source-loss-control distinction are documented in Section 4.10 and the retained worker sources. This package does not claim to reproduce the source paper's original benchmarks.
