# Preprocessing Pipeline: Old vs New

A detailed description of the LOS/NLOS preprocessing pipeline for the Rice Argos (TVT) massive-MIMO dataset. It covers:
- how the **old** pipeline (paper Section 3.3.1 and the original `preprocessing.ipynb`) worked and what went wrong;
- how the **new** pipeline (`preprocess_raw.ipynb`) works, step by step;
- how the new pipeline **prevents the loss of information** ("data shrink");
- what was **added**, what is **still missing**, and how each old problem was **solved**.

Related documents:

| Document | Contents |
|---|---|
| `../dataset_description/README.md` | the dataset: source, setup, file layout, clusters, recording conditions |
| `AUDIT.md` | every constant of the new pipeline checked against the raw data |
| `../paper_vs_new_preprocessing/README.md` | verification of the new output and the changes needed in the manuscript |
| `README.md` | how to run the notebook and which files each model uses |
| `data_exploration/subcarrier_view.ipynb` | plots and tables of the raw data per subcarrier |

---

## 1. Summary

| | Old pipeline | New pipeline |
|---|---|---|
| Input | `Pilot_Samples` (frames, 72, 1120, 2) | same file |
| What one value means | average \|I + jQ\| of one radio chain over all 1120 samples | **CSI amplitude** of the measured user on one antenna and one subcarrier |
| Values per frame | 72 | **3328** (64 antennas × 52 subcarriers); a compact 52-value form for the CNN-LSTM |
| Frequency information | none | 52 subcarriers |
| Signal used | all 5 slots mixed (user, other antennas, calibration node, noise, padding) | only the measured user's 64 pilot samples |
| Phase calibration | no | yes |
| Normalisation | z-score **per file, test files included** | **per-antenna** normalisation + z-score fitted on **training clusters only** |
| Frames | 54,825 | 53,016 (−3.3%, deliberate) |
| Verified | no | matches the dataset authors' CSI within **4e-6** |
| Output size | about 15 MB | about **6.3 GB** (full); about 11 MB (compact 52-value) |
| SVM / RF / CNN-LSTM on the paper's split | 82.2% / 53.8% / 91.0% | 99.4% / 99.9% / 100% |

---

## 2. The raw data in brief

Each raw file `raw_traces/channe14_<cluster>.hdf5` holds `Pilot_Samples` with shape **(frames, 72, 1120, 2)**:

- **72** = radio chains of the base station (18 boards × 4). The dataset authors' code maps **64** of them onto the 8 × 8 antenna array.
- **1120** = time samples per frame and radio chain = **5 slots × 224 samples**:
  - slots 0–3 are pilots from the 4 mobile antennas (one is the measured user, one the calibration node);
  - slot 4 is noise only.
- Inside each pilot slot, the **802.11 long training symbol (LTS)** starts at about sample 47 and lasts 64 samples.
- **2** = I and Q (int16).

Only **64 of the 1120 samples** of each radio chain (about 6%) are the measured user's pilot. A 64-point FFT of those samples gives the channel on 64 frequency bins, **52 of which are data subcarriers**.

Full details: `../dataset_description/README.md`.

---

## 3. The old pipeline

### 3.1 Steps

From the paper (Section 3.3.1, Eqs. 1–4) and the original notebook:

```python
for each cluster file:
    for each chunk of 200 frames:
        chunk = Pilot_Samples[i:i+200]               # (200, 72, 1120, 2)
        csi = chunk[..., 0] + 1j * chunk[..., 1]       # (200, 72, 1120)   Eq. 1
        amplitude = np.abs(csi)                        # (200, 72, 1120)   Eq. 2
        amplitude = np.mean(amplitude, axis=2)         # (200, 72)         Eq. 3
    processed = concatenate(chunks)                    # (frames, 72)
    processed = (processed - processed.mean()) / (processed.std() + 1e-8)   # Eq. 4, per file
    np.save(<cluster>.npy, processed)
```

Then:
- SVM / RF used each 72-value frame;
- CNN-LSTM / CNN-Transformer used windows of 50 frames with stride 10, shape (50, 72);
- training sequences were split **randomly 80:20** for validation.

### 3.2 Problems

| # | Problem | Effect |
|---|---|---|
| **P1** | **All 1120 samples averaged**, without separating the 5 slots | each value mixes the measured user, three other antennas, the calibration node, the noise slot and padding. About 94% of the averaged samples are not the user's pilot. |
| **P2** | **No FFT** | no subcarriers at all. The main physical LOS/NLOS difference, **frequency selectivity**, is lost. |
| **P3** | Magnitude taken **before** any processing | the channel can only be recovered from the complex samples. Once the magnitude is averaged, it can't be undone. |
| **P4** | The 72 values described as **"72 OFDM subcarriers"** | they are 72 **radio chains**, 8 of which aren't part of the 8 × 8 array. The paper's data model was wrong. |
| **P5** | **No phase calibration**, no antenna selection | each radio chain's own offsets stay in the data |
| **P6** | **z-score per file, test files included** | test statistics leak into preprocessing, and each cluster is rescaled to mean 0 / std 1 differently, which changes what the values mean between files |
| **P7** | The **receive-gain difference** (35 dB LOS vs 65 dB NLOS) and the **user's position** stay in the features | the 72 averages mostly measure gain, power and position, not propagation |
| **P8** | Gain-settling frames and the **empty last frame** kept | noisy and invalid frames in the data |
| **P9** | **Random 80:20 validation split** of overlapping windows | neighbouring windows share 40 of their 50 frames, so validation is almost a copy of training and early stopping is misled |
| **P10** | No record of which cluster each sample came from | only one fixed split can be evaluated; generalisation to other environments can't be measured |

### 3.3 Consequences

| Model (paper's split) | Old result | Reason |
|---|---|---|
| SVM | 82.2% (LOS recall 73%) | features reflect gain and power, not the channel |
| RF | **53.8%** (69% of LOS frames predicted as NLOS) | the same; the paper attributed it to "hand-crafted feature distributions" |
| CNN-LSTM | 91.0% | temporal patterns of signal level only |

---

## 4. The new pipeline

### 4.1 Overview

```
Pilot_Samples (frames, 72, 1120, 2)  int16
  │ 1. user's 64 LTS samples per radio chain (slot 0 LOS / slot 3 NLOS, from sample 47)
  │ 2. 64-point FFT, × known LTS                          → 64 bins
  │ 3. remove 12 empty bins (guard + DC)                   → 52 subcarriers
  │ 4. 64 of 72 radio chains (8 × 8 array)                 → (frames, 64, 52) complex
  │ 5. phase calibration with the calibration node         → CSI
  │ 6. skip first 200 frames, drop last (empty) frame
  │ 7. amplitude |H|, per-antenna normalisation            → (frames, 64, 52)
  │ 8. feature form: full 3328 (default) | 52 | 64 | 116
  │ 9. quality check (NaN / inf / empty frames)
  │10. save each cluster BEFORE z-score (+ complex CSI)
  │11. z-score fitted on training clusters only
  │12. frame datasets (SVM / RF)
  │13. 50-frame windows inside each cluster (CNN-LSTM / CNN-Transformer)
  └14. save, verify against the dataset authors' CSI
```

### 4.2 Step by step (`preprocess_raw.ipynb`)

| Notebook step | What it does | Shape after | Why |
|---|---|---|---|
| 1 Setup | paths and settings: `FEATURE`, `NORM`, `WINDOW`, `STRIDE`, `SKIP_FIRST`, `CHUNK`, `SAVE_CSI`, train/test clusters | — | all choices in one place |
| 2 Inspect a raw file | prints fields and attributes (`num_bs_ant`, `samples_per_user`, …) | — | documents the input |
| 3 Raw sample layout | slot layout, LTS offset, empty bins, antenna mapping | — | explains every constant (checked in `AUDIT.md`) |
| 4 CSI extraction (`extract_csi`) | for each 500-frame chunk and for the measured and calibration slots: LTS samples → FFT → × LTS → remove 12 bins → 64 antennas; then calibration H = H_meas / (H_cal / \|H_cal\|) | (frames, 64, 52) complex64 | turns raw samples into CSI (fixes P1–P5) |
| 5 Try on one cluster | runs Step 4 on NLOS 2 | (3853, 64, 52) | quick check |
| 6 Amplitude, normalisation, features (`frame_features`) | \|H\|; `NORM`: `"antenna"` (default) / `"frame"` / `"none"`; `FEATURE`: `"full"` (3328, default) / `"subcarrier"` (52) / `"antenna"` (64) / `"both"` (116) | (frames, 3328) | removes gain and position effects (fixes P7), keeps the frequency shape |
| 7 Quality check | removes NaN, inf and all-zero frames (none found) | unchanged | data integrity |
| 8 Process all 9 clusters | Steps 4–7 for every file; records the mean \|H\| before normalisation | 9 arrays | — |
| 9 Save each cluster | `clusters/<cluster>_features.npy` (**before z-score**), `clusters/<cluster>_csi.npy` (complex), `index.json` | per cluster | allows a separate z-score per split (fixes P10) |
| 10 z-score | mean/std per feature from the **training clusters only**, applied to all | unchanged | no test leakage (fixes P6) |
| 11 Frame datasets | `X/y/groups_{train,test}_frames.npy`, written straight to disk | (29487, 3328) / (23529, 3328) | SVM / RF input |
| 12 Sequences | 50-frame windows, stride 10, **inside each cluster** | (2926, 50, 3328) / (2336, 50, 3328) | CNN-LSTM / CNN-Transformer input |
| 13 Save | `y_*`, `groups_*`, `scaler.npz`, `config.json` | — | reproducibility |
| 14 Final checks | finite values, counts per cluster, **comparison with `channel_data/`** | — | verification (max difference about 4e-6) |

### 4.3 Settings and their justification

| Setting | Value | Justification (details in `AUDIT.md`) |
|---|---|---|
| LTS offset | 47 | the true LTS start is sample 46–54. 47 stays inside the guard interval, so the amplitude is unaffected. |
| FFT size | 64 | 802.11 OFDM |
| Empty bins | 0–5, 32, 59–63 | 802.11 guard bands + DC. In the raw data they are 13–38 dB below the data subcarriers (the DC bin carries receiver DC offset). |
| Measured / calibration slot | LOS (0, 3), NLOS (3, 0) | dataset authors' code; confirmed by signal power and timing per slot |
| Antenna selection | 64 of 72 | dataset authors' 8 × 8 array mapping |
| `SKIP_FIRST` | 200 | receiver gain settles after about 50 frames, so 200 is a safe margin (dataset authors' choice) |
| Last frame | dropped | all zeros in every file |
| `NORM` | `"antenna"` | per-frame normalisation left the antenna pattern, which encodes the user's position: LOS 3 was classified 0% correctly. Per-antenna normalisation fixed it (100%). |
| `FEATURE` | `"full"` | no reduction; compact forms can be computed later |
| `WINDOW`, `STRIDE` | 50, 10 | as in the paper |
| Train / test clusters | LOS 1, 2, NLOS 1–3 / LOS 3, 4, NLOS 4, 5 | the paper's split, worked out from its Table 4 |

---

## 5. How the new pipeline prevents data shrink

### 5.1 What "data shrink" was in the old pipeline

The old pipeline reduced every frame from **161,280 numbers to 72**, and **none of the 72 represented the user's channel on any subcarrier**. The reduction itself wasn't the main problem. The problem was that the useful information was **averaged away** before it was extracted:

| Stage | Old: kept | Old: lost |
|---|---|---|
| Separate the user's pilot from the other slots | — | the identity of the signal (everything mixed) |
| FFT to subcarriers | — | **all frequency information** |
| Per-antenna channel | one average per radio chain | the channel itself (only a power level remains) |
| Phase | — | discarded |

### 5.2 How the new pipeline keeps the information

| Mechanism | What it keeps |
|---|---|
| **Slot separation**: only the user's 64 LTS samples are used | the user's channel, without mixing in other transmitters or noise |
| **FFT + × known LTS** before any magnitude | the channel on **every subcarrier** |
| **No averaging** with `FEATURE = "full"` | all **64 antennas × 52 subcarriers = 3328** amplitudes per frame |
| **`SAVE_CSI = True`** | the **complex CSI** (amplitude **and phase**) of every frame, so nothing about the channel is thrown away |
| **Per-cluster files before z-score** | the unscaled features, so any later normalisation or split is possible without going back to the raw data |
| **Feature options** (`"subcarrier"`, `"antenna"`, `"both"`) | compact forms are **derived from** the full data instead of replacing it |
| **Memory-mapped writing** | the full-size data can be saved (6.3 GB) without running out of memory |

### 5.3 Size at each stage (LOS cluster 1)

| Stage | Shape | Size |
|---|---|---|
| Raw `Pilot_Samples` | (8140, 72, 1120, 2) int16 | 2.63 GB |
| Old output | (8140, 72) float32 | **2.34 MB**: no channel information |
| New: complex CSI | (7939, 64, 52) complex64 | **211 MB**: the complete channel |
| New: amplitude features | (7939, 3328) float32 | **106 MB** |
| New: compact (CNN-LSTM) | (7939, 52) float32 | 1.7 MB: derived from the full features |

### 5.4 What the new pipeline still removes, and why

| Removed | Amount | Reason |
|---|---|---|
| First 200 frames per cluster | 1,800 frames | receiver gain settling |
| Last frame per cluster | 9 frames | all zeros |
| Samples outside the user's pilot | about 94% of raw samples | other antennas, calibration node, noise slot, padding: not the user's channel (the noise slot is kept implicitly, because it can always be re-read from the raw file) |
| 12 empty FFT bins | 12 of 64 | no channel information |
| 8 radio chains | 8 of 72 | not part of the 8 × 8 array (dataset authors' mapping) |
| Absolute level (per-antenna normalisation) | — | reflects gain and position, not LOS/NLOS. It remains available in the complex CSI. |

**Frames removed: 1,809 of 54,825 (3.3%).** Every other removal discards data that isn't the user's channel.

| Frames per cluster | LOS 1 | LOS 2 | LOS 3 | LOS 4 | NLOS 1 | NLOS 2 | NLOS 3 | NLOS 4 | NLOS 5 | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| Raw (old) | 8,140 | 8,159 | 8,112 | 8,101 | 6,081 | 4,054 | 4,058 | 4,057 | 4,063 | 54,825 |
| New | 7,939 | 7,958 | 7,911 | 7,900 | 5,880 | 3,853 | 3,857 | 3,856 | 3,862 | 53,016 |

---

## 6. What is new in the pipeline

| # | Addition | Purpose |
|---|---|---|
| 1 | **CSI extraction** from raw I/Q: slot selection, LTS window, FFT, × LTS | real channel per subcarrier |
| 2 | **Removal of the 12 empty bins** | only data subcarriers |
| 3 | **Antenna selection** (64 of 72) following the dataset authors | the actual 8 × 8 array |
| 4 | **Phase calibration** with the calibration node | consistent phase across radio chains |
| 5 | **Skipping gain-settling frames** and the **empty last frame** | clean data |
| 6 | **`NORM` option**, with **per-antenna normalisation** as default | removes gain and position effects |
| 7 | **`FEATURE` option** (full, subcarrier, antenna, both) | full data or compact forms |
| 8 | **Quality check** for NaN / inf / empty frames | data integrity |
| 9 | **Per-cluster files before z-score** + `index.json` | evaluation on any split (80 splits used) |
| 10 | **Complex CSI saved** (`SAVE_CSI`) | phase and physical analysis (K-factor, eigenvalues, SNR tests) |
| 11 | **z-score on training clusters only** (`scaler.npz`) | no test leakage |
| 12 | **Cluster name for every sample** (`groups_*.npy`) | grouped validation, per-cluster results |
| 13 | **Windows cut inside each cluster** | no window spans two environments |
| 14 | **Memory-mapped writing** of large arrays | full resolution within about 4 GB of RAM |
| 15 | **`config.json`** with all settings | reproducibility |
| 16 | **Verification** against `channel_data/` | the extraction is proven correct (about 4e-6) |
| 17 | **`AUDIT.md`**: every constant checked on the raw data | justification for each value |

**Added on the model side** (in the model notebooks, using these outputs):
- **evaluation on all 80 splits** (20 leave-one-pair-out + 60 paper-style), with the z-score refitted for every split;
- **time-block / held-out-cluster validation** instead of a random split of overlapping windows;
- **compact 52-subcarrier input** for the CNN-LSTM, computed from `clusters/*_features.npy`.

---

## 7. How each old problem is solved

| Old problem | New solution | Evidence |
|---|---|---|
| P1: all slots mixed | only the measured user's LTS samples | slot roles confirmed by power and timing (`AUDIT.md`) |
| P2: no frequency information | FFT, 52 subcarriers | frequency selectivity: LOS 0.08–0.10 vs NLOS 0.17–0.48 |
| P3: magnitude before processing | FFT on complex samples first | CSI matches the dataset authors' within 4e-6 |
| P4: "72 subcarriers" | 72 radio chains → 64 antennas × 52 subcarriers | dataset attributes and authors' code |
| P5: no calibration | phase calibration | as in the dataset authors' processing |
| P6: z-score per file with test data | training-only z-score, refitted per split | scaler recomputed independently: exact match |
| P7: gain and position in the features | per-antenna normalisation | LOS 3: 0% → 100%; test std 1.05 → 0.73 (closer to training) |
| P8: invalid frames | first 200 and last frame removed | last frame all zeros in all 9 files |
| P9: leaky validation | time blocks with a gap, or held-out clusters (model notebooks) | CNN-LSTM: validation 99.7% vs test 85% before; now held-out estimate ≤ test |
| P10: one split only | cluster names + per-cluster files | 80-split evaluation for SVM, RF and CNN-LSTM |

### Effect on results

| Model | Old (paper's split) | New (paper's split) | New: leave-one-pair-out (20) | New: paper-style (60) |
|---|---|---|---|---|
| SVM | 82.2% | 99.4% | 90.9% ± 9.1% | 90.1% ± 7.1% |
| RF | 53.8% | 99.9% | 94.3% ± 8.2% | 90.4% ± 10.9% |
| CNN-LSTM | 91.0% | 100% (fixed version) | 99.6% ± 0.7% | 96.0% ± 6.2% |

---

## 8. What is still missing or limited

| # | Limitation | Why it matters | Possible remedy |
|---|---|---|---|
| 1 | **SNR difference not corrected** (about 38 dB LOS vs 19 dB NLOS, from the receive gain) | models can partly use noise level; an SVM tested on LOS data with noise added down to 19 dB drops to 85–96% | noise augmentation or SNR equalisation in training |
| 2 | **Interference spur kept** at k = −13 (also −14, −12) and a smaller one at k = +6 | stronger in NLOS recordings; the RF put 54% of its importance on it | optional exclusion of these subcarriers (kept by decision) |
| 3 | **Phase not used as a feature** | possible extra information (e.g. delay, angle) | phase-based features from `clusters/*_csi.npy` |
| 4 | **Only the first LTS copy used** | the second copy could add about 3 dB SNR | average both copies |
| 5 | **Antenna selection inherited** from the dataset authors | the 8 dropped radio chains aren't documented by the authors | keep all 72 as an option |
| 6 | **No real timestamps** (`FrameCompleteTime` all zeros) | the duration of a 50-frame window can't be verified | — |
| 7 | **`SKIP_FIRST = 200` is conservative** | gain settles after about 50 frames, so about 1,350 usable frames are dropped | `SKIP_FIRST = 50` |
| 8 | **Per-antenna normalisation removes absolute power** | power can carry real information (e.g. path loss), but here it is confounded with gain | keep both normalised and absolute features for comparison |
| 9 | **`FEATURE = "both"` can't be used with `NORM = "antenna"`** | the 64 antenna averages become constant | use `"full"` or `"subcarrier"` |
| 10 | **Compact 52-value form averages over antennas** | spatial information is lost in that form (kept in the full form) | PCA option in the CNN-LSTM code |
| 11 | **One measurement campaign, outdoor only** | cross-environment results are limited to unseen clusters of one site | a second dataset (see `../additional_datasets/README.md`) |
| 12 | **Labels per cluster only** | no per-frame ground truth | — |

---

## 9. Verification

| Check | Result |
|---|---|
| CSI of all 9 clusters vs the dataset authors' `channel_data/` | max difference 1e-6 to 6e-6 (float32 rounding) |
| Features = per-antenna-normalised \|CSI\| | difference ≤ 7.6e-6 |
| Scaler fitted on training clusters only | exact |
| Frame datasets: values, labels, cluster names | exact |
| All 5,262 windows = 50 consecutive frames of the right cluster | exact |
| No cluster in both train and test | yes |
| No NaN / inf | yes |

---

## 10. Running the pipeline

1. Open `preprocess_raw.ipynb` (any kernel with `h5py` and `numpy`) and **Run All**: about 30 seconds, peak RAM about 4 GB.
2. To run in Colab, uncomment the Drive lines in Step 1 and set `RAW_DIR` / `OUT_DIR`.

**Outputs** (`output_raw_full/`, about 6.3 GB):

| File | Shape | Use |
|---|---|---|
| `X_{train,test}_frames.npy` | (29487 / 23529, 3328) | SVM, RF (paper's split) |
| `X_{train,test}_seq.npy` | (2926 / 2336, 50, 3328) | CNN-LSTM, CNN-Transformer (paper's split) |
| `y_*.npy`, `groups_*.npy` | labels, cluster names | all models |
| `clusters/<cluster>_features.npy` | (frames, 3328), before z-score | evaluation on all splits |
| `clusters/<cluster>_csi.npy` | (frames, 64, 52) complex64 | physical analysis, SNR tests |
| `scaler.npz`, `config.json`, `clusters/index.json` | metadata | reproducibility |

Load large files with `np.load(path, mmap_mode="r")`. The 3328 values are antenna-major: `x.reshape(-1, 64, 52)` gives (antenna, subcarrier).

**Variants:**

| Goal | Settings |
|---|---|
| Compact data (52 values per frame) | `FEATURE = "subcarrier"`, `SAVE_CSI = False`, `OUT_DIR = "output_raw"` |
| Paper-like absolute amplitude | `NORM = "none"` |
| Per-frame normalisation (first version) | `NORM = "frame"` |
| Keep more frames | `SKIP_FIRST = 50` |
