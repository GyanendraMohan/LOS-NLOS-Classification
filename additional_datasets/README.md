# Additional Datasets for LOS/NLOS Validation and Cross-Dataset Comparison

The paper currently uses one dataset: the Rice Argos massive-MIMO dataset (outdoor, 9 clusters). Reviewers are likely to say that "cross-environment generalisation" is only tested within a single measurement campaign. This document lists public datasets that can be added to:

1. validate the models on other environments, hardware and frequency bands;
2. compare against published LOS/NLOS results;
3. run a **true cross-dataset test** using the physical features in `../channel_correlation_analysis/`.

---

## 1. Summary

| # | Dataset | Signal type | Environment | LOS/NLOS label | Fit |
|---|---|---|---|---|---|
| 0 | Rice Argos (current) | Massive-MIMO CSI, 64 ant × 52 sc, 2.4 GHz | Outdoor campus, 9 clusters | per cluster | baseline |
| 1 | KU Leuven Ultra Dense Indoor MaMIMO | Massive-MIMO CSI, 64 ant × 100 sc, 20 MHz | Indoor | per setup (URA LoS / URA NLoS) | **Tier 1: best match** |
| 2 | DICHASUS (cf0x / cf1x) | Distributed massive-MIMO CSI | Industrial hall | LoS and NLoS channels | Tier 1 |
| 3 | eWINE UWB LOS/NLOS | UWB channel impulse response (CIR) | 7 indoor environments | per sample | **Tier 2: standard benchmark** |
| 4 | Intel 5300 Wi-Fi HAR (LOS/NLOS) | Wi-Fi CSI | 3 indoor environments | per environment | Tier 3: supporting only |
| 5 | Wallhack1.8k | Wi-Fi CSI (ESP32-S3) | Office, LOS and through-wall | per recording condition | Tier 3: supporting only |
| 6 | DeepMIMO | Ray-traced (simulated) channels | Many scenarios | per user (exact) | optional, for ablations |

---

## 2. Dataset details

### 2.1 KU Leuven Ultra Dense Indoor MaMIMO CSI Dataset (Tier 1, recommended)

- **Authors:** S. De Bast, S. Pollin (KU Leuven, WaveCORE)
- **Access:** IEEE DataPort, DOI 10.21227/nr6k-8r78
- **Setup:** 64-antenna base station; 4 single-antenna users moved by CNC XY-tables; CSI on 100 subcarriers over 20 MHz.
- **Size:** 252,004 samples per setup.
- **Setups:** URA LoS, **URA NLoS**, ULA LoS, DIS LoS.
- **Why use it:**
  - It has the same kind of data as Argos (antennas × subcarriers), so the preprocessing and physical features apply with minimal changes.
  - It is **indoor**, which addresses the gap between the paper's "Wi-Fi / indoor" framing and the outdoor Argos data.
- **Caveats:**
  - There is only one NLOS setup, and labels are per setup.
  - Compare **URA LoS vs URA NLoS** so the array geometry doesn't confound the result.
  - Split train and test by spatial region (for example, grid quadrants), not at random.
  - The sources disagree on the grid area (1.25 m × 1.25 m vs 9 m²). Check the documentation.

### 2.2 DICHASUS (Tier 1)

- **Provider:** University of Stuttgart, distributed massive-MIMO channel sounder.
- **Access:** dichasus.inue.uni-stuttgart.de and DaRUS (CC BY 4.0).
- **Relevant subsets:**
  - `dichasus-cf0x`: distributed antennas, industrial environment (Arena2036), day 1. Contains LoS and NLoS.
  - `dichasus-cf1x`: same environment, day 2.
  - `dichasus-ca0x` / `cb0x`: industrial, LoS areas (possible LOS-only reference).
- **Features:** ground-truth positions (from GNSS, tachymeter or LiDAR), and a `spec.json` describing the file format.
- **Why use it:** a different site, hardware and type of environment (industrial). This gives a strong test of generalisation.
- **Caveats:** check how LoS/NLoS is marked. It may need to be worked out from antenna positions and geometry. Cite Euchner et al., WSA 2021.

### 2.3 eWINE UWB LOS/NLOS Data Set (Tier 2: standard benchmark)

- **Access:** GitHub `ewine-project/UWB-LOS-NLOS-Data-Set`
- **Hardware:** Decawave DWM1000 UWB.
- **Environments (7):** two offices, a small apartment, a small workshop, a kitchen with living room, a bedroom, a boiler room.
- **Size:** 3000 LOS + 3000 NLOS samples per environment, **42,000 in total**.
- **Content:** measured range plus the channel impulse response (CIR) and related features, with a LOS/NLOS label for each sample.
- **Why use it:**
  - It is the most widely used benchmark for LOS/NLOS classification, so you can compare directly with published work.
  - Every environment has both classes, so leave-one-environment-out works cleanly: **7 folds, each with a two-class test set**.
- **Caveats:**
  - Single antenna, UWB (not Wi-Fi or MIMO). Spatial features such as the eigenvalue share don't apply.
  - Use features from the time and frequency domain instead: K-factor-like statistics, delay spread, kurtosis and amplitude variation.
  - Check the repository README for the citation and licence.

### 2.4 Intel 5300 Wi-Fi HAR dataset in LOS and NLOS environments (Tier 3)

- **Source:** *Data in Brief*, 2020, "A dataset for Wi-Fi-based human activity recognition in line-of-sight and non-line-of-sight indoor environments".
- **Size:** 3000 trials (30 subjects × 5 experiments × 20 trials).
- **Environments:** 1 and 2 are LOS; 3 is NLOS.
- **Caveat:** each environment has only one class, so **the environment and the label are the same.** A classifier could learn "which room" instead of "LOS or NLOS". Use it only as supporting evidence and state this limitation.

### 2.5 Wallhack1.8k (Tier 3)

- **Authors:** J. Strohmayer, M. Kampel (TU Wien). This dataset belongs to ref [23] in the paper.
- **Access:** Zenodo; PyTorch loader at GitHub `StrohmayerJ/wallhack1.8k`.
- **Content:** 1,806 CSI amplitude spectrograms (about 4 s each, 400 packets at 100 Hz) plus raw packet data.
- **Folders:** `LOS/BQ`, `LOS/PIFA`, `NLOS/BQ`, `NLOS/PIFA`. BQ = biquad antenna, PIFA = printed inverted-F antenna.
- **Why use it:** cheap Wi-Fi hardware, already cited, and allows a test across two antenna types.
- **Caveats:**
  - It was designed for activity recognition (no presence / walking / walking with arm-waving). LOS/NLOS is a recording condition, not a per-sample label.
  - The published baselines (92.0 ± 3.5% BQ, 86.8 ± 4.7% PIFA) are **activity-recognition** accuracies, not LOS/NLOS.

### 2.6 DeepMIMO (optional, simulated)

- **Content:** ray-traced channels for many scenarios, with an exact LOS/NLOS flag for each user.
- **Use for:** controlled ablations, for example varying antenna count, bandwidth, K-factor or scenario.
- **Caveat:** simulated data. Reviewers will want real measurements as the main evidence.

### 2.7 Also noted

- **CSI-Bench** (arXiv 2505.21866, 2025): a large in-the-wild Wi-Fi CSI dataset across several chipsets, under LoS and NLoS conditions. It was not checked whether it has per-sample LOS/NLOS labels.

---

## 3. Comparison protocol

### 3.1 Within-dataset evaluation (same protocol everywhere)

| Dataset | Test protocol | Folds |
|---|---|---|
| Argos | Leave-one-pair-out: 1 LOS + 1 NLOS cluster held out | 20 |
| KU Leuven | URA LoS vs URA NLoS, split by spatial region (grid quadrants) | 4 |
| DICHASUS cf0x/cf1x | Split by spatial region or by measurement day | per dataset |
| eWINE UWB | Leave-one-environment-out | 7 |

In every case:
- validate **by group** (cluster, region or environment), never with a random split of overlapping windows;
- fit normalisation statistics on the training folds only;
- report **mean ± std** of accuracy, F1 and AUC over folds (and over 3–5 seeds for the DL models), plus the worst fold.

### 3.2 Cross-dataset transfer (new contribution)

The raw inputs differ between datasets (64×52, 64×100, a single-antenna CIR), so the CNN models can't be applied directly across datasets. The **physical features don't depend on input size**:

| Feature | Argos | KU Leuven | DICHASUS | eWINE |
|---|---|---|---|---|
| Rician K-factor | ✓ | ✓ | ✓ | ✓ (from CIR taps) |
| Amplitude variation across frequency (CV_f) | ✓ | ✓ | ✓ | ✓ (from FFT of CIR) |
| Frequency correlation ρ_f | ✓ | ✓ | ✓ | ✓ |
| Temporal correlation ρ_t (an SNR indicator only on Argos; see `../preprocessing_v2/AUDIT.md`) | ✓ | ✓ | ✓ | depends on sampling |
| Eigenvalue share λ₁/Σλ | ✓ | ✓ | ✓ | ✗ (single antenna) |

Planned experiments:
1. **Argos (outdoor) → KU Leuven (indoor):** train on one, test on the other, using physical features.
2. **Argos + KU Leuven → DICHASUS:** train on two, test on the third.
3. **Massive MIMO → eWINE:** only the features that don't need multiple antennas (K, CV_f, ρ_f).

Most CSI papers can't show transfer across datasets. A positive result here would be a strong, novel contribution.

### 3.3 Results table template for the paper

| Model / Features | Argos (20-fold) | KU Leuven | DICHASUS | eWINE (7-fold) | Argos → KU Leuven |
|---|---|---|---|---|---|
| SVM: physical features | | | | | |
| RF: physical features | | | | | |
| SVM: amplitude features | | | | | — |
| RF: amplitude features | | | | | — |
| CNN-LSTM | | | | | — |
| CNN-Transformer | | | | | — |
| Published results (eWINE) | — | — | — | *from literature* | — |

Each cell: mean ± std of accuracy / F1 / AUC.

---

## 4. Recommended plan

| Priority | Action | Effect on the paper |
|---|---|---|
| 1 (minimum) | Add **KU Leuven** (URA LoS vs URA NLoS) | Second dataset, indoor, same kind of data |
| 2 | **Argos → KU Leuven** transfer using physical features | True cross-dataset generalisation (new contribution) |
| 3 | Add **eWINE** with leave-one-environment-out | Standard benchmark and direct comparison with published work |
| 4 (optional) | DICHASUS, Wi-Fi datasets (4, 5), DeepMIMO | Further support or ablations; state the caveats clearly |

## 5. Next steps

1. Download KU Leuven (IEEE DataPort, free account required) and/or eWINE (GitHub).
2. Write a loader for each dataset that outputs CSI as (frames, antennas, subcarriers), or the CIR for eWINE.
3. Reuse `../channel_correlation_analysis/compute_channel_metrics.py` to extract physical features.
4. Run the protocol in Section 3 and fill in the table in Section 3.3.

---

## Sources

- [eWINE UWB LOS/NLOS Data Set (GitHub)](https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set)
- [Ultra Dense Indoor MaMIMO CSI Dataset (IEEE DataPort)](https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset)
- [KU Leuven massive MIMO measurement dataset paper (arXiv 2211.05527)](https://ar5iv.labs.arxiv.org/html/2211.05527)
- [DICHASUS datasets (University of Stuttgart)](https://dichasus.inue.uni-stuttgart.de)
- [DICHASUS cf0x: distributed antennas, industrial environment](https://b2find.eudat.eu/dataset/29bb054f-fdf2-5793-a354-1fcd06d4bfcb)
- [Wi-Fi HAR dataset in LOS and NLOS indoor environments (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7704290/)
- [Wallhack1.8k (Zenodo)](https://zenodo.org/records/13950918)
- [Wallhack1.8k paper (arXiv 2401.01388)](https://arxiv.org/abs/2401.01388v1)
- [CSI-Bench (arXiv 2505.21866)](https://arxiv.org/html/2505.21866v1)
