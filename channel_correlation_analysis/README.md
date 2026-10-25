# Physical Characterisation of LOS/NLOS Channels Using Vector Correlation

This analysis supports the paper *Robust CSI-Based Wireless Channel Condition Classification With Cross-Environment Generalization*. It shows that LOS and NLOS channels in the Argos dataset differ in measurable, physically meaningful ways. Those differences explain why the classifiers can tell the two apart.

- **Data:** `../tvt_open_source/channel_data/processed_*.hdf5`. Each file holds `mob_csi` with shape (64 antennas, frames, 52 subcarriers), complex and calibrated. It is identical to the CSI that `../preprocessing_v2/preprocess_raw.ipynb` extracts from `raw_traces/`.
- **Script:** `compute_channel_metrics.py`

```bash
python compute_channel_metrics.py --data_dir ../tvt_open_source/channel_data
```

---

## 1. Concept: the channel as a vector

In each frame, the channel seen by the 64-antenna base station on one subcarrier is a complex vector:

$$\mathbf{h} = [h_1, h_2, \dots, h_{64}]^T \in \mathbb{C}^{64}$$

### 1.1 Inner product = correlation

$$\mathbf{u}^H\mathbf{v} = \sum_{i} u_i^*\, v_i$$

This measures how similar two signal vectors are. The **conjugate** transpose (H) is required because CSI is complex. With the plain transpose, **u** = [1, j] would give **u**ᵀ**u** = 0, which is wrong.

### 1.2 Norm and energy

$$\|\mathbf{u}\|^2 = \mathbf{u}^H\mathbf{u} = \sum_i |u_i|^2 \;=\; \text{energy}$$

The energy is ‖**u**‖². The norm ‖**u**‖ is its square root.

### 1.3 Normalised correlation coefficient

$$\rho = \frac{|\mathbf{u}^H\mathbf{v}|}{\|\mathbf{u}\|\,\|\mathbf{v}\|}, \qquad 0 \le \rho \le 1$$

- Dividing by the norms removes the energy, so ρ does not depend on received power. It **does** depend on SNR: noise alone limits ρ to about SNR/(SNR+1).
- Taking the magnitude |·| removes any common phase rotation, such as an oscillator offset between frames.
- 0 ≤ ρ ≤ 1 by the Cauchy–Schwarz inequality. Geometrically, ρ = cos θ, where θ is the angle between the two vectors.
- No mean is subtracted. This is the signal-processing correlation (cosine similarity), not Pearson correlation.

### 1.4 Spatial covariance matrix

$$\mathbf{R} = E[\mathbf{h}\mathbf{h}^H] \in \mathbb{C}^{64\times64}$$

- **Diagonal** $R_{ii} = E[|h_i|^2]$: the energy at antenna *i*.
- **Off-diagonal** $R_{ij} = E[h_i h_j^*]$: the correlation between antennas *i* and *j*.
- **Eigenvalues** $\lambda_1 \ge \lambda_2 \ge \dots$: how the energy is spread over spatial directions. One dominant eigenvalue means one dominant propagation path.

### 1.5 Physical expectation

| | LOS | NLOS |
|---|---|---|
| Propagation | one strong direct path | many reflected / diffracted / scattered paths |
| Spatial structure | close to a single plane wave → **h** is nearly rank-1 | paths add up differently at each antenna |
| Across frequency | flat | frequency-selective fading |
| Over time | stable | fluctuates as the paths interfere |

---

## 2. Metrics

| Symbol | Metric | Definition | Expected |
|---|---|---|---|
| λ₁/Σλ | Dominant-eigenvalue share | λ₁ / Σλ of **R**, estimated over 50-frame windows with each **h** power-normalised | LOS → 1 |
| ρ_t | Temporal correlation | ρ(**h**ₜ, **h**ₜ₊₁), consecutive frames, across 64 antennas | LOS higher |
| ρ_f | Frequency correlation | ρ(**h**ₖ, **h**ₖ₊₁), adjacent subcarriers, across 64 antennas | LOS higher |
| K | Rician K-factor | direct-path power ÷ scattered power. Moment method: γ = Var(\|h\|²)/E[\|h\|²]², K = √(1−γ) / (1−√(1−γ)), over 50-frame windows | LOS higher |
| τ_rms | RMS delay spread | from the power-delay profile (IFFT over 52 subcarriers, 50 ns resolution) | LOS lower |
| CV_f | Amplitude variation | std/mean of \|h\| across subcarriers | LOS lower |

Every metric except τ_rms is a ratio, so none of them depend on absolute received power.

---

## 3. Results

| Cluster | λ₁/Σλ | ρ_t | ρ_f | K (dB) | τ_rms (ns) | CV_f |
|---|---|---|---|---|---|---|
| LOS 1 | 0.983 | 0.999 | 0.999 | 21.0 | 228 | 0.136 |
| LOS 2 | 0.961 | 0.999 | 0.999 | 18.2 | 225 | 0.165 |
| LOS 3 | 0.995 | 0.999 | 0.999 | 24.9 | 257 | 0.082 |
| LOS 4 | 0.990 | 0.999 | 0.999 | 22.0 | 239 | 0.095 |
| NLOS 1 | 0.821 | 0.980 | 0.972 | 9.1 | 255 | 0.386 |
| NLOS 2 | 0.847 | 0.974 | 0.933 | 8.8 | 318 | 0.480 |
| NLOS 3 | 0.892 | 0.979 | 0.978 | 10.7 | 251 | 0.279 |
| NLOS 4 | 0.937 | 0.981 | 0.978 | 11.8 | 235 | 0.288 |
| NLOS 5 | 0.820 | 0.978 | 0.976 | 7.9 | 251 | 0.355 |

### Class ranges

| Metric | LOS range | NLOS range | Separated? |
|---|---|---|---|
| **K (dB)** | 18.2 – 24.9 | 7.9 – 11.8 | **Yes. 6.4 dB gap; strongest separator** |
| **CV_f** | 0.082 – 0.165 | 0.279 – 0.480 | **Yes** |
| **λ₁/Σλ** | 0.961 – 0.995 | 0.820 – 0.937 | **Yes** |
| ρ_t | 0.999 | 0.974 – 0.981 | Yes, but **explained entirely by SNR** (see the note below). Not a physical feature. |
| **ρ_f** | 0.999 | 0.933 – 0.978 | **Yes** |
| τ_rms (ns) | 225 – 257 | 235 – 318 | No. Ranges overlap; 20 MHz bandwidth is too coarse |

Five of the six metrics separate all 9 clusters with no overlap. That includes the 4 test clusters the models never see during training (LOS 3, LOS 4, NLOS 4, NLOS 5).

### Important: SNR lines up with the label (see `../preprocessing_v2/AUDIT.md`)

The base station used 35 dB receive gain for LOS and 65 dB for NLOS. The resulting SNR per subcarrier is **about 38 dB (LOS) vs about 19 dB (NLOS)**. Noise lowers every correlation-type metric, so each metric was compared with the value noise alone would produce:

| Metric | NLOS value if only noise were present | Observed NLOS | Interpretation |
|---|---|---|---|
| ρ_t | ≈ 0.987 | 0.977 – 0.987 | **Fully explained by noise.** Drop it. |
| ρ_f | ≈ 0.987 | 0.933 – 0.978 | partly noise, partly real |
| λ₁/Σλ | ≈ 0.988 | 0.820 – 0.937 | **mostly real spatial spread** |
| K (dB) | ≈ 15 – 16 | 7.9 – 11.8 | **mostly real fading** |
| CV_f | ≈ 0.08 | 0.279 – 0.480 | **mostly real frequency selectivity** |

So λ₁/Σλ, K and CV_f carry real multipath information beyond SNR. ρ_t does not. To make the claim rigorous, use noise-corrected metrics (R − σ²I; subtract σ² before computing K, with σ² from the noise slot) and/or an SNR-equalised comparison.

---

## 4. How this strengthens the paper

### 4.1 Physical justification
The paper currently reports accuracy without showing *why* LOS and NLOS should be separable. These results show the classes differ in direct-path dominance (K), spatial concentration (λ₁/Σλ) and frequency selectivity (CV_f), as propagation theory predicts.

### 4.2 Ruling out the gain / SNR explanation
LOS and NLOS were recorded with different receive gains (35 vs 65 dB) and therefore different SNR (≈ 38 vs ≈ 19 dB). A model could in principle separate the classes by power or noise level alone.
- **Power:** all metrics are normalised, so they don't depend on received power.
- **SNR:** for λ₁/Σλ, K and CV_f, the NLOS values lie far beyond what noise alone would produce (table above), so multipath structure is the main cause.
- **To make this rigorous:** report noise-corrected metrics and an SNR-equalised test (add noise to LOS down to ≈ 19 dB).

> *Suggested sentence (after the SNR-equalised test confirms it):* "The LOS/NLOS separability persists in power-normalised, noise-corrected metrics such as the Rician K-factor and the dominant-eigenvalue share of the spatial covariance, and under SNR equalisation, confirming that the discrimination arises from multipath structure rather than received signal strength or noise level."

### 4.3 An interpretable baseline
These metrics can serve as a 4-value feature vector per window for SVM/RF (ρ_t is excluded because it only reflects SNR):

| Model | Input |
|---|---|
| SVM / RF | 116 amplitude features (current) |
| **SVM / RF** | **4 physical features (λ₁/Σλ, ρ_f, K, CV_f), noise-corrected** |
| CNN-LSTM / CNN-Transformer | 50 × 116 sequences |

- **If this baseline matches the DL models:** you have a lightweight, explainable classifier.
- **If the DL models clearly win:** you have evidence that they learn more than these summary statistics.

### 4.4 Support for the cross-environment claim
The metric ranges hold across all 9 spatially separate clusters. That gives a physical reason why a model trained on some clusters should generalise to unseen ones.

### 4.5 Links to the literature
- Li et al. [17] used the Rician K-factor and statistical CSI features for NLOS identification. These results can be compared directly.
- The dataset paper (Du & Sabharwal [26]) is about channel correlation, so the analysis fits the data.

### 4.6 Suggested placement in the manuscript
1. **New subsection 3.x, "Physical Characterisation of LOS/NLOS Channels":** equations 1.3, 1.4 and the K-factor; the table in Section 3; one figure (box plots of K and λ₁/Σλ per cluster).
2. **Section 4:** add the physical-feature SVM/RF baseline as an extra row in the results comparison.
3. **Section 4:** one paragraph on 4.2 (independence from power and SNR), plus the gain/SNR difference stated in the dataset description.
4. **Abstract and conclusion:** one sentence each on physical interpretability.

---

## 5. Caveats

- The values in Section 3 are **cluster averages**. Values for individual windows overlap more, so check the per-window distributions (box plots) before claiming perfect separability.
- ρ_t is limited by noise (SNR), not by the channel. See the SNR note in Section 3.
- The K-factor is estimated over 50-frame windows of a moving user, so it includes slow (large-scale) variation as well as small-scale fading.
- τ_rms is limited by the 20 MHz bandwidth (about 50 ns resolution). Report it as not discriminative here rather than leaving it out.
- λ₁/Σλ is computed on every 8th subcarrier and every 4th window to keep the run time short.

---

## 6. Next steps

1. Compute these metrics **per window** (50 frames, stride 10, the same windows as the DL models) and plot their distributions for each cluster.
2. Train SVM/RF on the 5 physical features with the same cluster-wise split, and report accuracy, F1 and AUC.
3. Add the figure and table to the manuscript (Section 4.6).
