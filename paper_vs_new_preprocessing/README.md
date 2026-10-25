# Paper vs New Preprocessing: Comparison and Verification

This document compares the preprocessing described in the paper *Robust CSI-Based Wireless Channel Condition Classification With Cross-Environment Generalization* (Sections 3.1, 3.3.1 and 3.3.2) with the new full-resolution pipeline in `../preprocessing_v2/preprocess_raw.ipynb`. It also records the verification of the new pipeline's output.

Related documents:
- `../paper_review_findings/README.md`: review of the paper (issues 5.1–5.4 are referenced below)
- `../preprocessing_v2/AUDIT.md`: check of every constant against the raw data, plus the SNR finding
- `../channel_correlation_analysis/README.md`: physical analysis of LOS/NLOS
- `../additional_datasets/README.md`: other datasets for validation

---

## 1. Summary

| | Paper | New pipeline |
|---|---|---|
| What is measured | average \|I/Q\| of raw time samples (roughly the received signal level per radio channel) | **real CSI** per antenna and subcarrier |
| Values per frame | 72 | **3328** (64 antennas × 52 subcarriers) |
| Normalisation | global z-score **per file**, including test files | per-antenna normalisation + per-feature z-score from **training clusters only** |
| Verified | no | **yes**: matches the dataset authors' CSI within 4e-6 |
| Output size | about 15 MB | **6.33 GB** |

---

## 2. Unchanged from the paper

| Item | Paper | New pipeline |
|---|---|---|
| Dataset | Rice Argos massive-MIMO dataset, `raw_traces/`, 9 clusters | same |
| Input field | `Pilot_Samples` only | same |
| Labels | LOS = 1, NLOS = 0, from the file name | same |
| Train clusters | 2 LOS + 3 NLOS | same: LOS 1, LOS 2, NLOS 1, NLOS 2, NLOS 3 |
| Test clusters (unseen) | 2 LOS + 2 NLOS | same: LOS 3, LOS 4, NLOS 4, NLOS 5 |
| Sliding window | window 50, stride 10, separately for each file | same |
| Inputs for the models | frames → SVM / RF; sequences → CNN-LSTM / CNN-Transformer | same |
| Loading in chunks | yes | yes (500 frames per chunk) |
| Complex values, then amplitude | Z = I + jQ, A = \|Z\| | same operations, applied to the CSI |
| z-score formula | (x − μ) / (σ + 1e-8) | same formula; computed differently (Section 3) |

The paper never names the clusters. The split was worked out exactly from its Table 4 (support 16,213 LOS / 8,120 NLOS) and its sequence counts (3,027 / 2,416). See `../paper_review_findings/README.md`, Section 2.

---

## 3. What changed

| Step | Paper | New pipeline | Reason |
|---|---|---|---|
| **Signal measured** | \|I + jQ\| of the **raw time-domain samples** | **CSI:** pilot (LTS) samples → 64-point FFT → × known LTS → channel per subcarrier | The paper's feature is roughly the received signal level, not CSI (issue 5.2). |
| **Samples used per frame** | all 1120: 4 user slots, the noise slot and padding | only the **64 LTS samples of the measured user** (offset 47 in that user's slot) | 1120 = 5 slots × 224. The other samples belong to other mobile antennas, the calibration node, noise or padding. |
| **Meaning of "72"** | called "72 OFDM subcarriers" (wrong) | 72 = **base-station radio channels**. After processing: **64 antennas × 52 subcarriers** | Issue 5.1. 802.11 has 64 FFT bins, of which 52 carry data and pilots. |
| **Subcarriers** | not extracted | 12 empty bins removed (guard bins 0–5 and 59–63, DC bin 32), leaving **52** | IEEE 802.11a/g layout; confirmed by energy in the raw data |
| **Antennas** | all 72 channels | **64 antennas** (order reversed; 8 channels dropped, following the dataset authors' array mapping) | 8 × 8 array as in the dataset paper |
| **Calibration** | none | phase calibration: H = H_meas / (H_cal / \|H_cal\|) | Removes phase offsets between radios. The amplitude is unchanged. |
| **Reduction** | mean over 1120 samples → **72 values** | **none: all 3328 values kept** (`FEATURE = "full"`) | keeps all spatial and frequency information |
| **Phase** | discarded | not used in the features, but **kept** in the saved complex CSI | available for later analysis |
| **Per-antenna normalisation** (`NORM = "antenna"`) | none | each antenna's 52 subcarriers ÷ that antenna's mean | Removes the receive-gain difference (**35 dB LOS vs 65 dB NLOS**) **and** the location-dependent pattern across antennas. With per-frame normalisation only, the SVM classified LOS cluster 3 0% correctly; with per-antenna, 100% (single split). |
| **z-score statistics** | one global mean/std **per file**, test files included | mean/std **per feature**, fitted on **training clusters only**, then applied to all | the paper's method leaks test information (issue 5.3) |
| **Frames dropped** | none mentioned | first 200 per cluster (gain settling) + the last frame (all zeros) | the last frame is empty in all 9 files; power settles within about 50 frames |
| **Quality check** | "no issues found" (not shown) | explicit check for NaN, inf and all-zero frames (none found) | — |
| **Cluster tracking** | none | the cluster name is saved for every sample (`groups_*.npy`) | needed for grouped validation and leave-one-pair-out |
| **Per-cluster files** | one `.npy` per cluster, already z-scored | features **before z-score** + complex CSI for every cluster | each leave-one-pair-out fold can refit its own scaler |
| **Saving large arrays** | in memory | written straight to disk (memory-mapped `.npy`) | full resolution needs about 6 GB on disk; peak RAM is about 4 GB |
| **Verification** | none | compared with the dataset authors' `channel_data/` | shows the extraction is correct |

---

## 4. Effect on the numbers

| Quantity | Paper | New pipeline | Difference |
|---|---|---|---|
| Frames (all clusters) | 54,825 | 53,016 | −1,809 (201 per cluster) |
| Values per frame | 72 | 3,328 | ×46 |
| Train frames (SVM/RF) | 30,492 * | 29,487 | −1,005 |
| Test frames (SVM/RF) | 24,333 (16,213 LOS / 8,120 NLOS) | 23,529 (15,811 LOS / 7,718 NLOS) | −804 |
| Train sequences | 3,027 | 2,926 (1,580 LOS / 1,346 NLOS) | −101 |
| Test sequences | 2,416 (1,613 LOS / 803 NLOS) | 2,336 (1,573 LOS / 763 NLOS) | −80 |
| Sequence shape | (50, 72) | (50, 3328), or reshaped to (50, 64, 52) | — |
| Output size | about 15 MB | 6.33 GB | — |

\* Worked out from Table 2: 8,140 + 8,159 + 6,081 + 4,054 + 4,058. The paper doesn't state it directly.

### Frames per cluster

| Cluster | Paper (Table 2) | New | Split |
|---|---|---|---|
| LOS 1 | 8,140 | 7,939 | train |
| LOS 2 | 8,159 | 7,958 | train |
| LOS 3 | 8,112 | 7,911 | test |
| LOS 4 | 8,101 | 7,900 | test |
| NLOS 1 | 6,081 | 5,880 | train |
| NLOS 2 | 4,054 | 3,853 | train |
| NLOS 3 | 4,058 | 3,857 | train |
| NLOS 4 | 4,057 | 3,856 | test |
| NLOS 5 | 4,063 | 3,862 | test |

---

## 5. Why the paper's data was so small, and the new data is not

Per frame, the paper went from 72 × 1120 × 2 raw numbers to 72 averages, about 1120 times smaller. The new pipeline keeps the full channel:

| Stage (LOS 1) | Shape | Size |
|---|---|---|
| Raw `Pilot_Samples` | (8140, 72, 1120, 2) int16 | 2.63 GB |
| Paper: mean over 1120 samples | (8140, 72) float32 | 2.34 MB |
| New: user's pilot → FFT → 52 subcarriers × 64 antennas (complex) | (7939, 64, 52) complex64 | 211 MB |
| New: amplitude features | (7939, 3328) float32 | 106 MB |

The raw file is much larger than the CSI because about 94% of the raw samples are not the measured user's pilot.

---

## 6. Verification of the new output

All files in `../preprocessing_v2/output_raw_full/` were checked independently against the raw-data pipeline and the dataset authors' processed CSI (`tvt_open_source/channel_data/`).

| # | Check | Result |
|---|---|---|
| 1 | Complex CSI of all 9 clusters vs `channel_data/` | ✅ max difference 1e-6 to 6e-6 (float32 rounding) |
| 1 | Features = frame-normalised \|CSI\| | ✅ relative difference ≤ 4e-6 (float32 rounding) |
| 1 | All values finite | ✅ no NaN or inf |
| 2 | Scaler fitted only on the 5 training clusters | ✅ |
| 3 | Frame datasets: values, labels, cluster names | ✅ exact match; training mean 0.000, std 1.000; test mean −0.001, std 1.050 |
| 4 | Sequences: every one of the 5,262 windows equals 50 consecutive frames of the correct cluster | ✅ exact match; labels and cluster names correct |
| 5 | No cluster in both train and test | ✅ |

---

## 7. Output files (`../preprocessing_v2/output_raw_full/`)

| File | Shape | Size | Used by |
|---|---|---|---|
| `X_train_frames.npy` | (29487, 3328) float32 | 393 MB | SVM, RF |
| `X_test_frames.npy` | (23529, 3328) float32 | 313 MB | SVM, RF |
| `X_train_seq.npy` | (2926, 50, 3328) float32 | 1.95 GB | CNN-LSTM, CNN-Transformer |
| `X_test_seq.npy` | (2336, 50, 3328) float32 | 1.55 GB | CNN-LSTM, CNN-Transformer |
| `y_*_frames.npy`, `y_*_seq.npy` | labels (LOS = 1, NLOS = 0) | small | all |
| `groups_*_frames.npy`, `groups_*_seq.npy` | cluster name per sample | small | grouped validation, leave-one-pair-out |
| `clusters/<cluster>_features.npy` × 9 | (frames, 3328), **before z-score** | 51–106 MB each, 0.71 GB total | leave-one-pair-out (refit scaler per fold) |
| `clusters/<cluster>_csi.npy` × 9 | (frames, 64, 52) complex64 | 103–212 MB each, 1.41 GB total | physical analysis, other features |
| `clusters/index.json` | label, frame count and file names per cluster | — | — |
| `scaler.npz` | training mean and std (3328 each) | 27 KB | applying the same scaling to new data |
| `config.json` | settings used | 1 KB | reproducibility |

| Group | Size |
|---|---|
| Sequence datasets | 3.50 GB |
| Per-cluster complex CSI | 1.41 GB |
| Frame datasets | 0.71 GB |
| Per-cluster features | 0.71 GB |
| Labels, cluster names, scaler, config, index | 0.003 GB |
| **Total** | **6.33 GB** (shown as 5.9 G by `du`, which uses binary units) |

**Usage notes:**
- Load the large files with `np.load(path, mmap_mode="r")` so they don't have to fit in RAM at once.
- The 3328 values are ordered antenna-major: `x.reshape(-1, 64, 52)` gives (antenna, subcarrier).
- A compact version (116 values per frame: 52 subcarrier means + 64 antenna means, about 160 MB in total) can be regenerated in about 30 seconds with `FEATURE = "both"`, `SAVE_CSI = False` and `OUT_DIR = "output_raw"`.

---

## 8. Things to know about the new data

1. **The SNR differs by class.** The base-station receive gain is 35 dB in every LOS file and 65 dB in every NLOS file. The resulting SNR per subcarrier is **about 38 dB (LOS) vs about 19 dB (NLOS)**. Per-frame normalisation removes the gain difference but **not the SNR difference**, so models may partly use noise level. An SNR-equalised test is recommended before claiming high accuracy (see `../preprocessing_v2/AUDIT.md`).
2. **NLOS has extreme amplitude peaks.** After per-frame normalisation (measured in the first version), the largest NLOS value is 30–42 times the frame mean, compared with about 3 times for LOS. These are real multipath and fading effects, but they can dominate SVM and the z-score. A log/dB transform (`20·log10(|H|)`) is worth trying for the classical models.
3. **SVM at full resolution is slow** (29k frames × 3328 features). RF, or the compact 116-feature version, is more practical for the classical baselines.
4. **Validation:** don't use a random 80:20 split on the sequences, because neighbouring windows share 40 frames. Hold out whole training clusters using `groups_train_seq.npy`.

---

## 9. Changes needed in the manuscript

1. **Section 3.1 (Dataset description):**
   - Replace "72 OFDM subcarriers" with "72 base-station radio channels, of which 64 antennas are used, with 52 subcarriers".
   - State the receive gain (35 / 65 dB) and SNR difference.
2. **Section 3.3.1 (Data pre-processing):** replace Eqs. 1–4 (complex → amplitude → mean over 1120 samples → global z-score) with:
   - LTS extraction and 64-point FFT;
   - removal of the 12 empty subcarriers;
   - antenna selection and phase calibration;
   - amplitude;
   - per-antenna normalisation;
   - per-feature z-score fitted on training clusters only.
3. **Tables 2 and 3:** update the frame counts (Section 4 of this document).
4. **Section 3.3.2 / 3.3.4:** update the sequence counts (2,926 / 2,336) and the input shape: (50, 3328) or (50, 64, 52) instead of 50 × 72.
5. **Tables 4–6 and Figures 4–7:** all results must be **re-run**. The current numbers come from the old features and the leaky normalisation.
6. **New content to add:**
   - the verification against the dataset authors' CSI;
   - the explicit train/test cluster names;
   - leave-one-pair-out results (20 folds, mean ± std);
   - the SNR-equalised control.

---

## 10. Next steps

1. Write the training notebook for SVM, RF, CNN-LSTM and CNN-Transformer on `output_raw_full/` (and/or the compact version).
2. Evaluate with leave-one-pair-out (1 LOS + 1 NLOS cluster held out per fold, 20 folds), refitting the scaler on each fold from `clusters/*_features.npy`.
3. Add the SNR-equalisation control and the physical-feature baseline (`../channel_correlation_analysis/`).
4. Update the manuscript as in Section 9.
