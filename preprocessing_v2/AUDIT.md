# Audit of `preprocess_raw.ipynb`

Every constant and design choice in the notebook was checked in two ways: against the dataset authors' code (`tvt_open_source/lib/argos_lib.py`, `lib/ota_lib.py`) and directly against the raw data in `raw_traces/`. Frames 300–2300 (or 1000–1100) of each file were used for these checks.

**Result:** the CSI extraction is correct, and every value is justified. However, the raw data shows a **20 dB SNR difference that lines up exactly with the LOS/NLOS label**. The notebook does not remove it, and it affects both the classifiers and the physical analysis (Section 3).

---

## 1. Constants

| Value in notebook | Source | Check on the raw data | Verdict |
|---|---|---|---|
| `Pilot_Samples` (frames, 72, 1120, 2) int16 | file | confirmed for all 9 files | ✓ |
| 1120 = 5 slots × 224 | attrs `samples_per_user = 224`, `num_mob_ant = 4` | slots 0–3 carry pilots; slot 4 is about 20 dB weaker (LOS −36 dB, NLOS −31 dB vs −12…−18 dB) and correlation with the LTS finds no pilot there, so it is **noise only** | ✓ |
| `LTS_OFFSET = 47` | `samps2csi(offset=47)` | true LTS start found by correlation: **46–54** depending on cluster and slot. Offset 47 is at most 1 sample late and at most 7 samples early, so the window stays inside the 32-sample cyclic prefix and the two repeated LTS. This adds only a linear phase ramp across subcarriers; **the amplitude is unchanged**. LTS 1 and LTS 2 extracted this way agree at ρ = 0.9997 (LOS) and 0.98–0.99 (NLOS). | ✓ |
| `FFT_SIZE = 64`, `pilot_length = 64` | attrs, 802.11 | — | ✓ |
| `LTS_FREQ` (802.11 LTS, ±1) | `lib/argos_lts.py` | multiplying by ±1 is the same as dividing by the known pilot | ✓ |
| `NULL_SUBCARRIERS` (0–5, 32, 59–63) | `samps2csi` | guard bins are 13–37 dB below the data subcarriers. DC (bin 32) still has energy (−5 to −6 dB) from DC offset / LO leakage, so it is correctly removed. The 52 kept bins are the strong ones. | ✓ |
| Scale `2**-15` | int16 full scale | only a constant factor; it has no effect after normalisation | ✓ |
| Measured / calibration slot: LOS (0, 3), NLOS (3, 0) | `array_config_gen` (`best_mea_idx`, `best_cal_idx`) | **Supported by the data.** In NLOS, slots 0/1 are strongest and their LTS starts at the same sample (46) in all 5 clusters, which fits a fixed calibration node near the base station. Slots 2/3 are weaker and arrive later (sample 49–54), which fits a user behind buildings. In LOS, slots 2/3 have the stable timing (48–49), so they are the calibration node. | ✓ |
| Choice of antenna within the user node (slot 3 vs 2 in NLOS) | authors' "best" choice | slot 3 is stronger in NLOS 1, 2 and 4, and within 0.5 dB in NLOS 3 and 5 | ✓ (reasonable) |
| Calibration `h_meas / (h_cal/|h_cal|)` | `extract_mob_CSI` | changes only the phase; the amplitude features are unaffected | ✓ |
| Antenna flip + `ANT_IDX` (64 of 72 kept) | `array_config_gen(remove_row=True)` | **The dropped channels are not faulty by power.** The selection maps the channels onto the authors' 8 × 8 array geometry. One kept channel (original index 60) is consistently about 10 dB weaker than its neighbours. | ⚠ inherited, not data-driven. Acceptable for reproducibility. Per-feature z-score absorbs fixed per-antenna offsets. |
| `SKIP_FIRST = 200` | `extract_mob_CSI(wind_lower=200)` | raw power settles within about 50 frames (77 dB → 75 dB), so 200 is a conservative margin and costs about 4% of frames | ✓ |
| Drop last frame | `wind_upper = max_frm − 1` | the last frame is all zeros in all 9 files | ✓ |
| Only LTS 1 used | `CSI_method="mob"` | LTS 2 is equally valid (ρ ≥ 0.98). Averaging both would gain about 3 dB SNR. | ✓ (optional improvement) |
| `CHUNK = 500` | memory | about 160 MB of int16 per chunk | ✓ |

## 2. Processing choices

| Choice | Justification | Verdict |
|---|---|---|
| Amplitude only | the paper's method uses amplitude; phase carries timing and CFO artefacts | ✓ |
| `NORM = "antenna"` (originally per-frame) | removes absolute scale, which partly reflects the **receive gain: 35 dB for every LOS file, 65 dB for every NLOS file** (attrs `bs_rx_gain`). Per-antenna normalisation also removes the location-dependent pattern across antennas (see `README.md`). | ✓ necessary, but not enough (Section 3) |
| `FEATURE = "both"` (52 + 64) | keeps frequency and spatial structure at low dimension | ✓ (design choice; `full` is available) |
| Quality check | removes NaN/inf/zero frames; none were found in any file | ✓ |
| z-score fitted on training clusters only, per feature | avoids leaking test-set information | ✓ |
| `WINDOW = 50`, `STRIDE = 10` | taken from the paper. Consecutive frames are continuous (no frame-to-frame correlation below 0.9 in 2000 checked frames per cluster), so windows are valid sequences. | ✓ |
| Train/test clusters | reproduce the paper's split exactly (Table 4 counts) | ✓ |
| Labels from file name | the only label available | ✓ |

**Unknown:** `FrameCompleteTime` is all zeros, so the real time between frames can't be checked. `frame_length = 50000` samples means 2.5 ms per frame nominally, so a 50-frame window is about 125 ms, but this cannot be verified from the data.

**Minor code note:** windows are cut after the quality check. If frames were ever removed from the middle of a cluster, a window could span the gap. No frames are removed for this dataset, so it has no effect now.

---

## 3. Main finding: SNR lines up with the label

Measured per subcarrier, using the noise slot (slot 4) passed through the same estimator:

| Cluster | `bs_rx_gain` | SNR (dB) | Noise-only ceiling on ρ_t = SNR/(SNR+1) | Observed ρ_t |
|---|---|---|---|---|
| LOS 1 | 35 | 39.2 | 0.9999 | 0.9995 |
| LOS 2 | 35 | 38.9 | 0.9999 | 0.9994 |
| LOS 3 | 35 | 37.1 | 0.9998 | 0.9995 |
| LOS 4 | 35 | 38.6 | 0.9999 | 0.9995 |
| NLOS 1 | 65 | 18.8 | 0.9870 | 0.9862 |
| NLOS 2 | 65 | 17.9 | 0.9841 | 0.9772 |
| NLOS 3 | 65 | 19.1 | 0.9880 | 0.9869 |
| NLOS 4 | 65 | 18.6 | 0.9863 | 0.9862 |
| NLOS 5 | 65 | 19.0 | 0.9877 | 0.9867 |

What this means:

1. **LOS ≈ 38 dB SNR and NLOS ≈ 19 dB SNR, with no overlap.** The weaker NLOS signal was boosted by 30 dB more receive gain, which raises the noise too. Frame normalisation removes the scale but **not the SNR**.
2. **ρ_t (frame-to-frame correlation) is explained entirely by noise.** The observed values match the noise-only ceiling, and they match the correlation between LTS 1 and LTS 2 *within the same frame*. The channel itself barely changes between frames; the drop in NLOS comes from noise. **ρ_t is therefore not evidence of multipath** and should not be used as a physical feature.
3. **The other physical metrics are mostly real, but noise contributes:**
   - **λ₁/Σλ:** noise alone would give about 0.988 for NLOS. The observed 0.82–0.94 is much lower, so spatial spread is real.
   - **K-factor:** noise alone would limit K to roughly SNR − 3 dB ≈ 15–16 dB for NLOS. The observed 8–12 dB is lower, so real fading is present.
   - **CV_f:** noise adds about 0.08 at 19 dB SNR. The observed NLOS values of 0.28–0.48 are mostly real frequency selectivity.
   - **ρ_f:** partly noise; the noise ceiling is about 0.987, and the observed NLOS range is 0.93–0.98.
4. **Classifier risk:** any model can partly separate the classes by **noise level** (for example, noisier amplitude patterns), even with frame normalisation. Near-perfect accuracy must be checked against this.

### Recommended controls

1. **SNR equalisation test (most important):** add complex Gaussian noise to the LOS CSI so its SNR drops to about 19 dB, the NLOS level. Then re-run the classifiers and the physical metrics. Any separation that survives cannot be explained by SNR.
2. **Noise-corrected metrics:** estimate the noise power σ² from slot 4 for every frame, then:
   - use R − σ²I for the eigenvalue share;
   - subtract σ² from E[|h|²] before computing K.
3. **Drop ρ_t** from the physical-feature set, or report it only as an SNR indicator.
4. **Optional:** average LTS 1 and LTS 2 (+3 dB SNR for both classes).
5. **Report the receive-gain difference (35 vs 65 dB) and the SNR difference in the paper's dataset description.** This is better than having a reviewer discover it.
