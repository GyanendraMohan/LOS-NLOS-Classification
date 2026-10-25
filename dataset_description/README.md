# Dataset Description and Data Loss in the Previous Preprocessing

This document describes the dataset used in the paper *Robust CSI-Based Wireless Channel Condition Classification With Cross-Environment Generalization*: where it comes from, how the raw files are organised, and what each part contains. It then explains **why the paper's original preprocessing lost most of the useful information**, and what the new pipeline keeps instead.

Related documents:
- `../preprocessing_v2/AUDIT.md`: every preprocessing constant checked against the raw data
- `../paper_vs_new_preprocessing/README.md`: step-by-step comparison and verification of the new pipeline
- `../preprocessing_v2/data_exploration/subcarrier_view.ipynb`: plots and tables of the raw data per subcarrier

---

## Part 1: Dataset description

### 1.1 Source

| Item | Value |
|---|---|
| Name | Massive MIMO Multi-User Channel Correlation Dataset (open access) |
| Authors | X. Du, A. Sabharwal (Rice University) |
| Reference | *Massive MIMO Channels with Inter-User Angle Correlation: Open-Access Dataset, Analysis and Measurement-Based Validation*, IEEE Trans. Vehicular Technology, 2021, DOI 10.1109/TVT.2021.3131606 |
| Platform | **Argos** massive-MIMO base station (Rice University, RENEW project) |
| Environment | **outdoor**, Rice University campus |
| Original purpose | studying how correlated the channels of different users are; not LOS/NLOS classification |
| Local copy | `../tvt_open_source/` |

### 1.2 Measurement setup

| Item | Value | Source |
|---|---|---|
| Frequency | 2.4 GHz ISM band, Wi-Fi channel 14 (about 2.484 GHz) | file attribute `channel = 14`; dataset code uses 2.4835 GHz |
| Bandwidth | 20 MHz | dataset paper |
| Waveform | IEEE 802.11 OFDM training symbols: 64-point FFT, **52 data subcarriers**, 312.5 kHz spacing | dataset code, raw data |
| Base station | **18 boards × 4 radios = 72 radio channels**; **64 of them form the 8 × 8 antenna array** used in the analysis | attributes `num_bs_nodes = 18`, `num_bs_ant = 72`; dataset code |
| Mobile side | **2 mobile nodes × 2 antennas = 4 mobile antennas**: one acts as the measured user, one as a calibration node | attributes `num_mob = 2`, `num_mob_ant = 4` |
| Calibration node | fixed, about 20 m from the base station; used to correct phase offsets between radios | dataset paper |
| Receive gain (base station) | **35 dB for all LOS recordings, 65 dB for all NLOS recordings** | attribute `bs_rx_gain` |

### 1.3 Clusters (environments)

The measurements are grouped into **9 clusters**. Each covers an area of about 4 m in diameter with more than 25 user positions. The user moved through the cluster while the base station recorded.

| Cluster | Class | Description |
|---|---|---|
| LOS 1–4 | LOS (label 1) | direct line of sight between user and base station |
| NLOS 1–5 | NLOS (label 0) | direct path blocked by campus buildings; signal arrives by reflection, diffraction and scattering |

The label comes from the file name. There is no per-frame label.

### 1.4 Folder contents

| Folder | Contents | Size | Used? |
|---|---|---|---|
| `raw_traces/` | 9 HDF5 files of raw I/Q samples, one per cluster | about 17.5 GB | **yes: the only input** |
| `channel_data/` | the dataset authors' processed CSI (`mob_csi`, 64 antennas × frames × 52 subcarriers, complex) | about 2.6 GB | only to **verify** the new pipeline (matches within 4e-6) |
| `data/` | 200 cached results of the authors' correlation simulations | about 0.2 GB | no (simulation, no LOS/NLOS content) |
| `lib/`, `*.py` | the dataset authors' processing and plotting code | small | as reference for the extraction steps |

### 1.5 Raw files (`raw_traces/channe14_<cluster>.hdf5`)

Example: `channe14_LOS_cluster1.hdf5` (2.65 GB).

| Field | Shape | Type | Contents | Used? |
|---|---|---|---|---|
| **`Pilot_Samples`** | (8140, 72, 1120, 2) | int16 | raw received samples: frames × radio channels × time samples × (I, Q) | **yes** |
| `RSSI` | (8140, 5, 72) | uint32 | received signal strength per slot and radio channel | no |
| `FrameCompleteTime` | (8140, 2) | uint64 | intended frame timestamps; **all zeros**, so no real timing is available | no |
| `mob_tx_gain` | (62, 4, 2) | int32 | mobile transmit-gain settings | no |

Main file attributes:

| Attribute | Value | Meaning |
|---|---|---|
| `num_bs_ant` | 72 | base-station radio channels |
| `num_bs_nodes` | 18 | base-station boards (4 radios each) |
| `num_mob` / `num_mob_ant` | 2 / 4 | mobile nodes / mobile antennas |
| `samples_per_user` | 224 | samples per slot (see 1.6) |
| `sample_dump` | 1120 | samples recorded per frame and radio channel |
| `pilot_length` | 64 | length of one pilot symbol |
| `frame_length` | 50000 | frame length in samples (2.5 ms at 20 MHz, nominal) |
| `bs_rx_gain` | **35 (LOS) / 65 (NLOS)** | base-station receive gain in dB |
| `mob_rx_gain` | 52 | mobile receive gain |

### 1.6 Structure of one frame

For every frame, each of the 72 radio channels records **1120 time samples**, divided into **5 slots of 224 samples**:

```
1120 samples = 5 slots × 224
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│  slot 0  │  slot 1  │  slot 2  │  slot 3  │  slot 4  │
│ mobile 1 │ mobile 1 │ mobile 2 │ mobile 2 │  noise   │
│ antenna A│ antenna B│ antenna A│ antenna B│ (nobody  │
│          │          │          │          │ transmits)│
└──────────┴──────────┴──────────┴──────────┴──────────┘
inside each pilot slot: guard interval | LTS (64) | LTS copy (64) | rest
                                       ^ starts at about sample 47
```

| Slot | LOS files | NLOS files |
|---|---|---|
| 0 | **measured user** | calibration node |
| 1 | other antenna of the same node | other antenna of the same node |
| 2 | other antenna of the same node | other antenna of the same node |
| 3 | calibration node | **measured user** |
| 4 | noise only | noise only |

The slot roles follow the dataset authors' code and are confirmed by the raw data (`AUDIT.md`): the calibration slots have the same timing in every cluster, as expected for a fixed node.

Each pilot slot carries the **802.11 long training symbol (LTS)**: 64 known samples, sent twice. A 64-point FFT of the LTS, divided by the known LTS values, gives the channel on each subcarrier. Of the 64 FFT bins, **52 are data subcarriers**. The other 12 are empty: 11 guard bins and the DC bin.

**Only 64 of the 1120 samples per radio channel are the measured user's channel** (about 6%). The rest belong to other antennas, the calibration node, the noise slot, the second LTS copy or padding.

### 1.7 Frames per cluster

| Cluster | Label | Raw frames | Frames after preprocessing | Sequences (window 50, stride 10) | Paper's split |
|---|---|---|---|---|---|
| LOS 1 | 1 | 8,140 | 7,939 | 789 | train |
| LOS 2 | 1 | 8,159 | 7,958 | 791 | train |
| LOS 3 | 1 | 8,112 | 7,911 | 787 | test |
| LOS 4 | 1 | 8,101 | 7,900 | 786 | test |
| NLOS 1 | 0 | 6,081 | 5,880 | 584 | train |
| NLOS 2 | 0 | 4,054 | 3,853 | 381 | train |
| NLOS 3 | 0 | 4,058 | 3,857 | 381 | train |
| NLOS 4 | 0 | 4,057 | 3,856 | 381 | test |
| NLOS 5 | 0 | 4,063 | 3,862 | 382 | test |
| **Total** | | **54,825** | **53,016** | **5,262** | |

Classes: LOS 31,708 frames (59.8%), NLOS 21,308 frames (40.2%), after preprocessing.

### 1.8 Recording conditions that matter for classification

These properties of the recordings line up with the LOS/NLOS label. A model can exploit them, so they must be reported:

| Property | LOS | NLOS | Effect |
|---|---|---|---|
| Base-station receive gain | 35 dB | 65 dB | absolute amplitude reflects a hardware setting; removed by per-antenna normalisation |
| SNR per subcarrier | **about 38 dB** | **about 19 dB** | noisier NLOS data; **not removed** by any normalisation |
| Interference spur at subcarrier k = −13 (also k = −14, −12) | +5 dB above the noise floor | **+16.5 to +18 dB** | a non-physical bump in NLOS frames; the RF put 54% of its feature importance on it |
| Smaller spur at k = +6 | +4 to +5 dB | none | appears only in the LOS recordings |
| Frequency selectivity (CV across subcarriers) | 0.08–0.10 | 0.17–0.48 | **the real physical LOS/NLOS difference** |

### 1.9 Limitations

- **One measurement campaign:** one site, one base station, one hardware setup. "Unseen environments" here means unseen clusters of the same campaign.
- **Outdoor only:** no indoor data.
- **Labels per cluster,** not per frame.
- **No usable timestamps:** the real time between frames can't be checked.

---

## Part 2: Why the previous preprocessing lost data

### 2.1 What the previous preprocessing did

The paper (Section 3.3.1) and the original notebook `preprocessing.ipynb` did this for each cluster file:

```
Pilot_Samples (frames, 72, 1120, 2)
  1. Z = I + jQ                         -> (frames, 72, 1120) complex
  2. A = |Z|                            -> (frames, 72, 1120)
  3. mean over the 1120 samples         -> (frames, 72)          72 values per frame
  4. z-score with this file's own mean/std
```

The 72 values were described as "72 OFDM subcarriers". They are actually the **72 base-station radio channels**.

### 2.2 How much was kept

| | Numbers per frame | Contents |
|---|---|---|
| Raw `Pilot_Samples` | 72 × 1120 × 2 = **161,280** | all slots, all radio channels |
| The measured user's channel in it | 64 antennas × 52 subcarriers = **3,328 complex values** | the CSI |
| **Previous preprocessing** | **72** | one average per radio channel |
| New preprocessing (full) | **3,328** amplitudes (+ complex CSI saved) | the CSI amplitude per antenna and subcarrier |
| New preprocessing (compact, CNN-LSTM input) | **52** | the CSI amplitude per subcarrier, averaged over antennas |

So the previous output was about 2,240 times smaller than the raw data per radio channel. More importantly, **none of the 72 values corresponded to the user's channel on any subcarrier.**

### 2.3 Where the information was lost, and why

| # | Step | What was lost | Why it matters |
|---|---|---|---|
| 1 | **Averaging over all 1120 samples** without separating the slots | the identity of the signal: the average mixes the **measured user, the other antennas, the calibration node, the noise slot and padding** | about 94% of the averaged samples are not the user's pilot, so the result is not the user's channel |
| 2 | **No FFT** | **all frequency information**: the 52 subcarriers never existed in the output | LOS vs NLOS differs mainly in **frequency selectivity**: a flat response (LOS) vs fading across subcarriers (NLOS). Averaging over time samples erases it. |
| 3 | Taking the magnitude before any processing | the relationship between samples needed to recover the channel (an FFT of magnitudes is meaningless) | the CSI can only be obtained from the complex samples |
| 4 | **"72 subcarriers"** | correct interpretation: 72 = radio channels, including 8 that are not part of the 8 × 8 array | the method and the paper's description were based on a wrong data model |
| 5 | No phase calibration, no antenna selection | consistency between radio channels | each radio's own gain and phase offsets stayed in the data |
| 6 | **z-score per file**, test files included | **the level differences between clusters** (each file rescaled to mean 0, std 1) and the separation of training and test data | test statistics leaked into preprocessing, and each cluster was rescaled differently |
| 7 | Gain-settling frames and the empty last frame kept | — (not a loss, but noisy data) | the first frames of each file are recorded while the gain settles; the last frame is all zeros |

### 2.4 What the 72 values really measured

Each of the 72 values was the **average received signal level of one radio channel over the whole frame**. That level depends mostly on:
- the **receive gain** (35 vs 65 dB),
- the **received power** of all transmitting antennas and the calibration node,
- the **position of the user** relative to each antenna.

None of these is a reliable LOS/NLOS feature in a new environment. The results show this:

| Model (paper's split) | Previous preprocessing (paper) | New preprocessing |
|---|---|---|
| SVM | 82.2% | 99.4% |
| Random Forest | **53.8%** (69% of LOS frames predicted as NLOS) | 99.9% |
| CNN-LSTM | 91.0% | 100% (fixed version) |

The paper explained the RF's 53.8% as "RF is based on hand-crafted feature distributions". The real cause was the input: the same RF reaches 99.9% on real CSI.

### 2.5 What the new preprocessing keeps

```
Pilot_Samples (frames, 72, 1120, 2)
  1. take only the measured user's 64 LTS samples (slot 0 for LOS, 3 for NLOS, from sample 47)
  2. 64-point FFT, divide by the known LTS          -> channel on 64 bins
  3. remove the 12 empty bins                        -> 52 data subcarriers
  4. select the 64 antennas of the 8 × 8 array
  5. phase calibration with the calibration node     -> complex CSI (frames, 64, 52)
  6. amplitude, then divide each antenna by its own mean (removes gain and position effects)
  7. z-score fitted on training clusters only
```

The result matches the dataset authors' own CSI (`channel_data/`) to within 4e-6.

### 2.6 What the new preprocessing still removes, deliberately

| Removed | Amount | Reason |
|---|---|---|
| First 200 frames of each cluster | 1,800 frames | receiver gain still settling (power settles after about 50 frames, so 200 is a safe margin) |
| Last frame of each cluster | 9 frames | all zeros |
| Samples outside the user's pilot | about 94% of the raw samples | other antennas, calibration node, noise slot, padding: not the user's channel |
| 12 empty FFT bins | 12 of 64 | guard bands and DC: no channel information |
| 8 radio channels | 8 of 72 | not part of the 8 × 8 array (dataset authors' mapping) |
| Phase (in the features) | — | carries timing artefacts; the complex CSI is still saved in `clusters/*_csi.npy` |
| Absolute level (per-antenna normalisation) | — | reflects the gain setting and the user's position, not LOS/NLOS |

**Total frames removed: 1,809 of 54,825 (3.3%).** Every other removal discards data that isn't the user's channel.

---

## References

1. X. Du, A. Sabharwal, "Massive MIMO Channels with Inter-User Angle Correlation: Open-Access Dataset, Analysis and Measurement-Based Validation," *IEEE Trans. Vehicular Technology*, 2021. DOI 10.1109/TVT.2021.3131606.
2. X. Zhang, L. Zhong, A. Sabharwal, "Directional Training for FDD Massive MIMO," *IEEE Trans. Wireless Communications*, vol. 17, no. 8, pp. 5183–5197, 2018. DOI 10.1109/TWC.2018.2838600.
3. IEEE 802.11a/g OFDM physical layer: 64-point FFT, 52 used subcarriers, long training symbol.
