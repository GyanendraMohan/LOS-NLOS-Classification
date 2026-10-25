# Review Findings: CSI-Based LOS/NLOS Classification Paper

**Paper:** *Robust CSI-Based Wireless Channel Condition Classification With Cross-Environment Generalization*
(M. Gulhane, B. K. Umrao, N. K. Gupta). File: `../Mridula_ITER_WiFi_Sensing.pdf`

**Dataset:** Rice Argos Massive MIMO Multi-User Channel Correlation dataset (Du & Sabharwal, IEEE TVT 2021). Folder: `../tvt_open_source/`

**Preprocessing code reviewed:** `~/Downloads/preprocessing.ipynb`

---

## 1. Datasets used

The paper uses **one dataset only**, the Argos dataset (ref [26]). All training and testing for SVM, RF, CNN-LSTM and CNN-Transformer comes from it.

- Other datasets in Related Work are only cited, not used in experiments. Examples: Jiang et al. [25] with 1.3M samples from 14 datasets, and the indoor WLAN data in [17]–[19].
- Labels come from the file names (`LOS_clusterX` → 1, `NLOS_clusterX` → 0). There is no external ground truth.
- The "unseen environments" are held-out clusters from the same campaign, recorded with the same base station, hardware and outdoor campus site. **There is no cross-dataset test.**

### Folders inside `tvt_open_source/`

| Folder | Contents | Used in paper? |
|---|---|---|
| `raw_traces/` (~17.5 GB, 9 files) | Raw I/Q radio samples per cluster | **Yes. Only input.** |
| `channel_data/processed_*.hdf5` | Original authors' processed CSI (`mob_csi`, 52 subcarriers, calibrated) | No |
| `data/_numerical_cache_*.hdf5` | Cached simulation results for the original paper's figures | No |
| `paper_figures/`, `plots/` | Figures from the original dataset paper | No |

### Fields inside each raw `.hdf5` file

| Key | Shape (NLOS cluster 5 example) | dtype | Used? |
|---|---|---|---|
| `Pilot_Samples` | (4063, 72, 1120, 2) | int16 | **Yes** |
| `RSSI` | (4063, 5, 72) | uint32 | No |
| `FrameCompleteTime` | (4063, 2) | uint64 | No |
| `mob_tx_gain` | (104, 4, 2) | int32 | No |

---

## 2. Train/test split (worked out from the paper's sample counts)

The paper does not name which clusters are used for training and which for testing. The counts in Table 4 and the sequence totals fix them exactly.

| File | Frames | Split | Sequences (win 50, stride 10) |
|---|---|---|---|
| `channe14_LOS_cluster1.hdf5` | 8140 | Train | 810 |
| `channe14_LOS_cluster2.hdf5` | 8159 | Train | 811 |
| `channe14_LOS_cluster3.hdf5` | 8112 | **Test** | 807 |
| `channe14_LOS_cluster4.hdf5` | 8101 | **Test** | 806 |
| `channe14_NLOS_cluster1.hdf5` | 6081 | Train | 604 |
| `channe14_NLOS_cluster2.hdf5` | 4054 | Train | 401 |
| `channe14_NLOS_cluster3.hdf5` | 4058 | Train | 401 |
| `channe14_NLOS_cluster4.hdf5` | 4057 | **Test** | 401 |
| `channe14_NLOS_cluster5.hdf5` | 4063 | **Test** | 402 |

The counts check out:
- **SVM/RF test set:** LOS 8112 + 8101 = 16213; NLOS 4057 + 4063 = 8120; total 24333. This matches Table 4.
- **DL test sequences:** 807 + 806 = 1613 LOS; 401 + 402 = 803 NLOS; total 2416. This matches.
- **DL training sequences:** 810 + 811 + 604 + 401 + 401 = 3027. This matches.
- **All frames:** 32512 LOS + 22313 NLOS = 54825. This matches Table 3.

**Recommendation:** name the training and test clusters explicitly in Section 3.3.2 / 4.1.

---

## 3. Preprocessing pipeline (paper Section 3.3.1 and the notebook)

1. Load `Pilot_Samples` in chunks of 200 frames.
2. Build complex values: `Z = I + jQ` (Eq. 1).
3. Take the amplitude: `A = |Z| = sqrt(I² + Q²)` (Eq. 2).
4. Average over the 1120 time samples, giving a (frames, 72) array (Eq. 3).
5. Apply z-score normalisation: `(x − mean) / (std + 1e-8)` (Eq. 4).
6. Save one `.npy` per cluster.
7. Feed the models:
   - **SVM / RF:** each 72-value frame is one sample.
   - **CNN-LSTM / CNN-Transformer:** sliding windows of 50 frames with stride 10, so each sample is 50 × 72. Windows are cut separately for each file and split.

---

## 4. Why the data becomes so small after preprocessing

Most of the reduction comes from step 4, the mean over 1120 time samples.

| Stage | Shape | dtype | Size (LOS cluster 1) |
|---|---|---|---|
| Raw `Pilot_Samples` | (8140, 72, 1120, 2) | int16 | ≈ 2.63 GB |
| Amplitude | (8140, 72, 1120) | float32 | ≈ 2.63 GB |
| **Mean over time samples** | **(8140, 72)** | float32 | **≈ 2.34 MB** |

Per frame, 72 × 1120 × 2 = 161,280 numbers become 72 numbers:

- Amplitude: the I and Q values (2) become 1, so ÷2.
- Mean over time: 1120 values become 1, so ÷1120.
- int16 → float32: each value takes twice the bytes, so ×2.
- Net effect: about **1120× smaller**.

The whole dataset goes from about 17.5 GB to about 15 MB. The notebook output confirms this: 1.17–2.35 MB per `.npy` file.

### Which dimensions are removed

The raw axis order is **(frames, 72, 1120, 2)**, not (frames, 1120, 72, 2). The notebook prints `Pilot_Samples (4063, 72, 1120, 2)`, which is why the code averages over `axis=2`.

| Step | Shape | Dimension removed | Reduction |
|---|---|---|---|
| Raw | (8140, 72, 1120, **2**) | — | — |
| Amplitude `sqrt(I² + Q²)` | (8140, 72, **1120**) | **I/Q pair (2 → 1)** | ÷2 |
| `np.mean(axis=2)` | (8140, **72**) | **1120 time samples → 1 average** | ÷1120 |

1. **The amplitude step** merges the real and imaginary parts (the last axis, size 2) into one value.
2. **The mean step** turns the 1120 time samples for each antenna into a single average. This step accounts for almost all of the reduction.

`np.mean(axis=2)` removes the 1120 axis completely, so the result is (8140, 72), not (8140, 72, 1). Each antenna in each frame is now one number instead of 1120 × 2 = 2240.

Those 1120 samples are the raw signal over time: the pilot, the per-user slots and the noise slot. Averaging them is what loses the real CSI per subcarrier (see 5.2).

### About the 500–700 MB figure

A figure of 500–700 MB is not the saved `.npy` files. It is likely Colab RAM used while processing chunks: one 200-frame chunk as complex128 is about 260 MB, plus temporary arrays.

---

## 5. Technical issues

### 5.1 "72" is antennas, not subcarriers
In `Pilot_Samples`, axis 72 is the **base-station antenna/radio channel** and axis 1120 is **raw time-domain samples**. See `tvt_open_source/lib/argos_lib.py` (`samps2csi`): its docstring gives the layout as `[Frame, BS Ant, Samples, I/Q]`. The paper is inconsistent:
- In Section 3.1, the CNN-Transformer description and Fig. 3, the 72 values are called "OFDM subcarriers". **This is wrong.**
- In Section 3.3.1, *m* = number of antennas. **This is correct.**

### 5.2 The features are not true CSI amplitude
Real CSI is obtained as follows (`argos_lib.py`, `samps2csi`):
1. Split the 1120 samples into per-user slots of 224 samples. The last slot is noise.
2. Take the 64-sample LTS (pilot) segments at offset 47.
3. Run a 64-point FFT and multiply by the known LTS.
4. Drop the null subcarriers, which leaves **52 data subcarriers**.

The paper instead averages |I/Q| over all 1120 raw samples. That window likely also covers the noise slot and non-pilot samples. The result is roughly the **average received signal level per antenna**, not CSI per subcarrier.

**Options:**
- (a) Describe the feature honestly, for example as "per-antenna mean received amplitude".
- (b) Compute proper CSI with `samps2csi`.
- (c) Use the ready-made `channel_data/processed_*.hdf5` (`mob_csi`).

### 5.3 Normalisation: the code does not match the paper, and it leaks test information
- The paper (Eq. 4) says the **global** mean and std are used.
- The notebook (`process_file_in_chunks`) computes mean and std **for each file separately**.

This has two consequences:
1. Each cluster is scaled independently. This removes overall level differences between clusters, which can change what the model learns.
2. The test clusters are normalised with their **own** statistics, so information from the test set enters preprocessing.

**Fix:**
1. Save the un-normalised amplitudes.
2. Compute mean and std on the training clusters only (LOS 1, LOS 2, NLOS 1, NLOS 2, NLOS 3).
3. Apply those values to every file.
4. Make the paper text match the code.

### 5.4 The model may be learning the measurement setup, not the channel
The LOS and NLOS recordings differ in hardware settings that line up exactly with the label (verified in `../preprocessing_v2/AUDIT.md`):
- **Receive gain:** `bs_rx_gain` is 35 dB in every LOS file and 65 dB in every NLOS file.
- **SNR per subcarrier:** about 38 dB for LOS vs about 19 dB for NLOS, with no overlap.
- **Calibration roles:** different calibration slots are used (`TVT_LOS_array` vs `TVT_NLOS_array`). The code comment "+20 dB + 15 dB attenuation" for NLOS most likely refers to attenuators on the calibration node; it does not say where they were fitted.

Each class has only 2 test clusters. A near-perfect result (99.9%) may therefore reflect the gain or SNR difference rather than LOS/NLOS propagation. Reviewers may raise this.

**Suggested checks:**
- Leave-one-cluster-out evaluation.
- Per-frame normalisation, to remove the absolute level. This does **not** remove the SNR difference.
- **SNR equalisation:** add noise to LOS CSI to reach about 19 dB SNR, then re-test.
- Proper subcarrier CSI features.

### 5.5 The generalisation claim is too broad
The title and abstract claim cross-environment generalisation for "smart indoor applications". However, the data is **outdoor**, comes from **one dataset**, and uses **one hardware setup**. Either:
- narrow the claim to "unseen clusters within one outdoor deployment", or
- add a second, independent (ideally indoor) LOS/NLOS CSI dataset for a cross-dataset test.

---

## 6. Problems in the manuscript text

### 6.1 Content from an unrelated paper
- **Algorithm 1 "AnchorZoneDiscovery"** (Section 3.4) belongs to a routing-protocol paper and has unresolved `Equation (??)` references.
- **The Conclusion** describes an "energy-efficient anchor-based geographical routing protocol (EAZR)". It must be rewritten for this paper.
- **Biography** ("figures/jpeg/naveen1.jpg …") is template residue.

### 6.2 Placeholders that were never filled in
- Fig. 2 caption: "....Provide suitable title....".
- Section 4.1: "A table needed".
- Section 3.2: "Past tenstances issue".

### 6.3 Inconsistent numbers

| Item | Where | Conflict |
|---|---|---|
| CNN-LSTM accuracy | Text: 96.52% | Table 5 and confusion matrix: 91% (1403/1613 LOS correct, 210 misclassified) |
| CNN-LSTM per-class metrics | Text: LOS recall 0.95, NLOS precision 0.91 | Table 5: LOS recall 0.87, NLOS precision 0.79 |
| CNN-LSTM misclassified LOS | Text: 84 samples (1529 correct) | Fig. 4: 210 samples (1403 correct) |
| CNN-Transformer AUC | Abstract: 0.94 | Results: ≈1.0 / 0.9999. (0.94 is the **RF** AUC) |
| CNN-LSTM AUC | Abstract: 0.99 | Fig. 5: 1.00 |
| CNN-Transformer errors | Text: "all 1613 LOS correct, 2 NLOS wrong" | Fig. 6: 4 LOS → NLOS and 2 NLOS → LOS (6 errors) |
| CNN-Transformer accuracy | Text: 99.92% ("only two samples misclassified") | Fig. 6 gives 2410/2416 = 99.75% |
| Table 6 | NLOS F1 printed as "100" | Should be 1.00 |
| SVM AUC | Text: 0.9998 | Not shown in a figure or table |
| Table 2 / Table 3 captions | Table 2 says "samples in dataset", Table 3 "samples in each file" | Captions are swapped |

### 6.4 Smaller issues
- In the Fig. 3 / Section 3.2 description, the CNN-LSTM is referred to as "flow chart diagram 2" but it is Fig. 3.
- Eq. 3 is written as `1/t Σ_{i=1}^{t} A_{n,m,t}`. The summation index should be the same symbol as the summed variable.
- In Section 3.1, the "52 subcarriers" (dataset paper) and "72 subcarriers" (raw traces) are mixed up. See 5.1.
- In Section 4.2, RF is described as "based on hand-crafted feature distributions". It was trained on the same 72-value amplitudes as SVM, with no hand-crafted features.
- Section 3.3.3 mentions a "Precision–Recall (PR) curve", but none is shown.
- Spelling: "Jupiter Notebook" should be "Jupyter Notebook"; "Quadrant" should be "Quadrature" (I/Q).

---

## 7. Priority action list

1. Remove the routing-paper content (Algorithm 1, Conclusion) and fill in all placeholders.
2. Reconcile every reported number with the actual confusion matrices and tables (Section 6.3).
3. Fix the "72 subcarriers" wording to "72 antenna channels", or switch to true 52-subcarrier CSI.
4. Fix normalisation to use training-set-only global statistics, and rerun all four models.
5. Name the train/test clusters explicitly.
6. Add leave-one-cluster-out results and/or a second dataset to support the generalisation claim.
7. Narrow the "indoor" claim, since the data is outdoor only.
