# Preprocessing v2: LOS/NLOS classification

This version replaces the original notebook (`preprocessing.ipynb`). It fixes issues 5.1–5.4 in `../paper_review_findings/README.md`.

## Files

| File | Purpose |
|---|---|
| `preprocess_raw.ipynb` | The preprocessing pipeline. Reads `raw_traces/` (raw I/Q) and extracts the CSI: LTS → FFT → 52 subcarriers → 64 antennas → calibration. Checked against the dataset authors' `channel_data/`: maximum difference about 4e-6. |
| `output_raw_full/` | **Main dataset** for all models (full resolution, about 6.3 GB) |
| `AUDIT.md` | Check of every constant in the notebook against the raw data, plus the SNR finding |
| `PIPELINE_README.md` | Detailed comparison of the old and new pipelines: data loss, new additions, limitations, how each old problem is solved |

A detailed comparison with the paper's preprocessing, including the verification results, is in `../paper_vs_new_preprocessing/README.md`.

## Run

Open `preprocess_raw.ipynb` and **Run All**. It needs `h5py` and `numpy`, takes about 30 seconds and peaks at about 4 GB of RAM.

To run in Colab, uncomment the Drive lines in Step 1 and set `RAW_DIR` / `OUT_DIR` to your Drive folders.

## Settings (Step 1)

| Setting | Default | Meaning |
|---|---|---|
| `FEATURE` | `"full"` | `"full"` = all 64 × 52 = 3328 amplitudes per frame. `"both"` = 116 averages (52 subcarrier + 64 antenna). `"subcarrier"` = 52. `"antenna"` = 64. |
| `SAVE_CSI` | `True` | also save the complex CSI (amplitude and phase) for every cluster |
| `NORM` | `"antenna"` | `"antenna"`: each antenna ÷ its own mean, which removes absolute power **and** the location-dependent pattern across antennas (recommended). `"frame"`: each frame ÷ its mean. `"none"`: absolute amplitude. With `"antenna"`, use `FEATURE = "full"` or `"subcarrier"`. |
| `WINDOW`, `STRIDE` | 50, 10 | sliding window, as in the paper |
| `SKIP_FIRST` | 200 | frames dropped at the start of each cluster (gain settling); the last frame (all zeros) is always dropped |
| `OUT_DIR` | `"output_raw_full"` | where outputs are written |

**Compact version:** to regenerate a small dataset, set `FEATURE = "subcarrier"` (52 values per frame), `SAVE_CSI = False` and `OUT_DIR = "output_raw"`, then Run All. `FEATURE = "both"` (116) needs `NORM = "frame"`, because per-antenna normalisation makes the 64 antenna averages constant.

## Pipeline

| Step | Paper | v2 |
|---|---|---|
| Input | `Pilot_Samples` (72, 1120, 2) raw I/Q | same file, but only the 64 LTS samples of the measured user |
| Signal | \|I + jQ\| of raw time samples | CSI: FFT → × known LTS → 52 subcarriers × 64 antennas, phase-calibrated |
| Reduction | mean over 1120 samples → 72 values | none: 3328 values per frame (default) |
| Normalisation | none | each antenna ÷ its own mean amplitude (default; `NORM = "antenna"`) |
| z-score | mean/std **per file**, including the test files | per-feature mean/std from the **training clusters only** |
| Split | cluster-wise | same clusters as the paper |
| Sequences | window 50, stride 10, per file | same |

Train clusters: LOS 1, LOS 2, NLOS 1, NLOS 2, NLOS 3.
Test clusters (unseen): LOS 3, LOS 4, NLOS 4, NLOS 5.

## Outputs (`output_raw_full/`)

| File | Shape | Size | Used by |
|---|---|---|---|
| `X_train_frames.npy` | (29487, 3328) | 393 MB | RF, SVM (see note) |
| `X_test_frames.npy` | (23529, 3328) | 313 MB | RF, SVM |
| `X_train_seq.npy` | (2926, 50, 3328) | 1.95 GB | CNN-LSTM, CNN-Transformer |
| `X_test_seq.npy` | (2336, 50, 3328) | 1.55 GB | CNN-LSTM, CNN-Transformer |
| `y_*.npy` | labels (LOS = 1, NLOS = 0) | small | all |
| `groups_*.npy` | cluster name per sample | small | grouped validation |
| `clusters/<cluster>_features.npy` | (frames, 3328), **before z-score** | 51–106 MB each, 0.7 GB total | leave-one-pair-out (refit scaler per fold) |
| `clusters/<cluster>_csi.npy` | (frames, 64, 52) complex64 | 103–212 MB each, 1.4 GB total | physical analysis, SNR tests |
| `clusters/index.json`, `scaler.npz`, `config.json` | metadata | small | reproducibility |

The counts are lower than in the paper (for example, 3027 → 2926 training sequences) because the first 200 frames and the last frame of each cluster are dropped.

The 3328 features are ordered antenna-major: `x.reshape(-1, 64, 52)` gives (antenna, subcarrier). Load large files with `np.load(path, mmap_mode="r")`.

## Which files for which model

| Use | Files |
|---|---|
| CNN-LSTM / CNN-Transformer, paper's split | `X_*_seq.npy` |
| RF, paper's split | `X_*_frames.npy` |
| SVM | RBF SVM on 3328 features is very slow. Use the compact version (`FEATURE = "both"`), PCA fitted on training data, or `LinearSVC`. |
| Leave-one-pair-out (20 folds) | `clusters/*_features.npy`: fit the z-score on each fold's training clusters, then build frames/windows. Do **not** use the ready-made `X_*` files here, because their scaler was fitted on the paper's fixed training clusters. |
| Physical features, SNR-equalisation test | `clusters/*_csi.npy` |

## Notes for training

- **Validation split:** do **not** use a random 80:20 split on the sequences. Neighbouring windows overlap by 40 frames, so validation windows would nearly duplicate training windows. Split by cluster instead, using `groups_train_*.npy`. For example, hold out one LOS and one NLOS training cluster.
- **Why per-antenna normalisation:** with `NORM = "frame"`, the SVM learned the location-dependent pattern across antennas and classified LOS cluster 3 0% correctly. Its antenna pattern correlates −0.39 to −0.47 with the other LOS clusters. With `NORM = "antenna"`, the same SVM reached 100% on LOS 3 and 99.4% overall (single split, analysis run).
- **Normalisation variants:** compare `NORM = "antenna"`, `"frame"` and `"none"`. If accuracy is only high **without** normalisation, the model is probably detecting received power or gain, not the propagation itself.
- **SNR lines up with the label** (about 38 dB LOS vs 19 dB NLOS), and no normalisation option removes it. Run an SNR-equalised test as well. See `AUDIT.md`.
- **NLOS has extreme amplitude peaks:** up to 30–42 times the frame mean, compared with about 3 times for LOS. Consider a log/dB transform for SVM.
- Mean |H| per cluster (in `config.json`) does not separate the classes cleanly. LOS 3 (1.53) is lower than NLOS 1, 3 and 5 (1.55–1.58).
