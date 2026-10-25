# Implementation Plan for the LOS/NLOS Journal Paper

The plan for turning the current work into a high-quality journal paper on **cross-environment LOS/NLOS classification from massive-MIMO CSI**. It lists every remaining task in order, with its goal, method, outputs and the criterion for calling it done.

**Working title:** *Cross-Environment LOS/NLOS Classification from Massive-MIMO CSI: Location-Invariant Normalisation and Environment-Invariant Learning*

**Target journals:** mid-tier SCI/SCIE (*IEEE Access*, *Sensors*, *Physical Communication*) as the safe target; upper-mid (*IEEE Sensors Journal*, *Computer Communications*) if Phases 4 and 7 succeed. See Phase 10.

Related documents:

| Document | Contents |
|---|---|
| `../dataset_description/README.md` | the Argos (TVT) dataset |
| `../preprocessing_v2/PIPELINE_README.md` | the preprocessing pipeline, old vs new |
| `../preprocessing_v2/AUDIT.md` | checks of every preprocessing constant; SNR and spur findings |
| `../paper_review_findings/README.md` | problems in the current manuscript |
| `../channel_correlation_analysis/README.md` | physical LOS/NLOS characterisation |
| `../additional_datasets/README.md` | candidate second datasets |

---

## 1. Rules that apply to every experiment

All results in the paper must follow these rules, so they can be compared.

| Rule | Detail |
|---|---|
| **Same splits** | the 20 leave-one-pair-out (`lopo`) and 60 paper-style splits, exactly as defined in the existing notebooks (Step "List all splits") |
| **No test leakage** | normalisation statistics, PCA, hyperparameters and epoch counts are chosen on **training clusters only** |
| **Validation** | held-out *training* clusters (inner folds), never random windows and never test clusters |
| **Same input definitions** | `full` = 3328 values (64 antennas × 52 subcarriers); `subcarrier` = 52 values; both per-antenna normalised |
| **Reporting** | mean ± std over splits, minimum, accuracy per held-out cluster, macro-F1, ROC-AUC; the paper's single split only as a secondary result |
| **Randomness** | deep models: 3 seeds (42, 43, 44) on all splits; report mean ± std across seeds |
| **Results format** | one `splits_<protocol>.jsonl` file per model and variant (same fields as now), so the statistics notebook can read everything |
| **Execution** | notebooks are written and smoke-tested, then **run by the user**; results are reviewed after each phase |

---

## 2. Current status

| Item | Status | Location |
|---|---|---|
| Preprocessing from raw I/Q, verified (4e-6) | ✅ done | `preprocessing_v2/preprocess_raw.ipynb` |
| Per-antenna normalisation (fixes location dependence) | ✅ done | same |
| SVM, 3328 inputs, 80 splits | ✅ done: lopo 90.9% ± 9.1%, paper-style 90.1% ± 7.1% | `preprocessing_v2/SVM/` |
| RF, 3328 inputs, 80 splits | ✅ done: 94.3% ± 8.2% / 90.4% ± 10.9% | `preprocessing_v2/RF/` |
| CNN-LSTM v1 (3328 inputs): environment overfitting | ✅ done: 87.9% ± 14.8% / 84.6% ± 10.5% | `preprocessing_v2/CNN_LSTM/results_v1_full_resolution/` |
| CNN-LSTM fixed (52 inputs, held-out-cluster epoch selection) | ✅ done: **99.6% ± 0.7% / 96.0% ± 6.2%** | `preprocessing_v2/CNN_LSTM/` |
| Data artefacts identified (gain, SNR, spurs) | ✅ documented | `AUDIT.md`, `data_exploration/` |
| Physical features (cluster level) | ✅ analysis done | `channel_correlation_analysis/` |
| Dataset and pipeline documentation | ✅ done | `dataset_description/`, `PIPELINE_README.md` |

---

## 3. Phases

Phases 1–3 make the existing results **fair and robust**. Phase 4 adds the **new method**. Phases 5–7 add **evidence**. Phases 8–10 produce the **paper**.

```
Phase 1  Fair baselines (SVM/RF on 52 inputs)                ┐
Phase 2  Robustness (spur, SNR, seeds)                        ├─ can start now, in parallel
Phase 6  Statistics notebook (reads all results)              ┘
Phase 3  CNN-Transformer (fixed approach)
Phase 8  Literature check  ──►  Phase 4  New method (environment-invariant learning + augmentation)
Phase 5  Physical characterisation per window
Phase 7  Second dataset (DICHASUS)                             (optional but valuable)
Phase 9  Manuscript
Phase 10 Reproducibility package and submission
```

---

### Phase 1: Fair baselines

**Goal:** compare all models on the **same input**. The fixed CNN-LSTM uses 52 values per frame, but SVM/RF used 3328, so the current comparison is not fair.

| Task | Method | Output | Done when |
|---|---|---|---|
| 1.1 SVM on 52 inputs | the existing SVM Part B with an `INPUT` setting (`full` / `subcarrier`), z-score per split | `SVM/results_52/splits_*.jsonl` | all 80 splits run |
| 1.2 RF on 52 inputs | the same for RF | `RF/results_52/splits_*.jsonl` | all 80 splits run |
| 1.3 (optional) SVM/RF on window features | mean and std of each of the 52 values over the 50-frame window (104 values), so the frame models also see temporal context | `*/results_52win/` | all 80 splits run |

**Decides:** whether the CNN-LSTM's advantage comes from the **model** or from the **input representation**. Both outcomes are publishable (Phase 9 adapts the claim).

---

### Phase 2: Robustness

**Goal:** show that the results don't depend on recording artefacts or on luck.

| Task | Method | Output | Done when |
|---|---|---|---|
| 2.1 Spur exclusion | remove subcarrier columns 12–14 (k = −14…−12) and column 31 (k = +6) from the input (52 → 48; 3328 → 3072); re-run the best SVM, RF and CNN-LSTM | `*/results_nospur/` | 80 splits per model; change in accuracy reported |
| 2.2 SNR-equalised test | add complex Gaussian noise to the LOS CSI so its SNR matches NLOS (about 19 dB); evaluate the trained models on it | table: accuracy at 38 / 25 / 19 / 15 dB LOS SNR | accuracy drop measured for each model |
| 2.3 Noise augmentation | during training, add noise to each frame at a random SNR (15–40 dB) | `CNN_LSTM/results_aug/` | 80 splits; compare with and without |
| 2.4 Seeds | CNN-LSTM (and later CNN-Transformer) with seeds 43 and 44 | `results_seed43/`, `results_seed44/` | mean ± std across seeds |

**Done when:** the paper can state how much accuracy changes without the spur subcarriers and under equal SNR.

---

### Phase 3: CNN-Transformer

**Goal:** complete the paper's four models with the same fixes as the CNN-LSTM.

| Task | Method | Output | Done when |
|---|---|---|---|
| 3.1 Architecture | the paper's design: Conv1D blocks → positional encoding → 2 Transformer encoder blocks → global average pooling → dense head; compact 52 input | `preprocessing_v2/CNN_Transformer/` | notebook + core module, as for the CNN-LSTM |
| 3.2 Training | epochs chosen on held-out training clusters; one process per split | `results/splits_*.jsonl` | 80 splits, 3 seeds |
| 3.3 (optional) full-resolution variant | 3328 input, to show whether it overfits like CNN-LSTM v1 | `results_v1/` | 80 splits |

---

### Phase 4: New method

**Goal:** a methodological contribution aimed at the main weakness: failure on particular unseen environments (LOS 2, NLOS 3, and splits that hold out LOS 1 + LOS 2).

**Proposed method:** an *environment-invariant CNN-LSTM with artefact-aware augmentation*.

| Task | Method | Output | Done when |
|---|---|---|---|
| 4.1 Environment-invariant training | treat each training cluster as a **domain**. Variant A: **domain-adversarial training** (an extra head predicts the cluster; a gradient-reversal layer removes cluster information from the features). Variant B: **Group DRO** (minimise the loss of the worst training cluster). | `CNN_LSTM_invariant/` | both variants on 80 splits |
| 4.2 Artefact-aware augmentation | SNR equalisation by noise injection, random subcarrier masking (including the spur band), random gain | combined with 4.1 | 80 splits |
| 4.3 Physics-informed input (optional) | add per-window physical features (CV across subcarriers, noise-corrected K-factor, spatial eigenvalue share; from Phase 5) to the dense head | variant of 4.1 | 80 splits |
| 4.4 Ablation | base model, + augmentation, + invariance, + physics features, all combined | ablation table | every row on all 80 splits |

**Success criterion:** higher **worst-split** accuracy and higher accuracy on the hard clusters than the fixed CNN-LSTM, with statistical significance (Phase 6), without losing mean accuracy.

**Before starting:** finish Phase 8, so the method is new for this problem.

---

### Phase 5: Physical characterisation per window

**Goal:** physical evidence that LOS and NLOS really differ, beyond the cluster averages already computed.

| Task | Method | Output | Done when |
|---|---|---|---|
| 5.1 Features per window | per 50-frame window: frequency selectivity (CV across subcarriers), **noise-corrected** Rician K-factor (noise power from the noise slot), spatial eigenvalue share (covariance minus noise) | `channel_correlation_analysis/window_features.npz` | computed for all 9 clusters |
| 5.2 Distributions | box plots per cluster and per class | figures for the paper | — |
| 5.3 Physical baseline | SVM/RF on the 3 physical features, 80 splits | `results_physical/` | 80 splits |
| 5.4 Correlation with errors | do the hard clusters (LOS 2, NLOS 3) lie between the classes in these features? | short analysis | — |

---

### Phase 6: Statistics notebook

**Goal:** one notebook that reads every `splits_*.jsonl` and produces all tables, tests and figures for the results section. It updates automatically as new results arrive.

| Analysis | Method |
|---|---|
| Mean ± std and **95% confidence interval** per model | bootstrap over splits (10,000 resamples) |
| Pairwise comparison | **paired Wilcoxon signed-rank test**, with **Holm correction** |
| Effect size | median paired difference and rank-biserial correlation |
| Correction for overlapping training sets | **corrected resampled t-test** (Nadeau & Bengio), reported alongside Wilcoxon |
| All models together | **Friedman test** + **Nemenyi** post-hoc, with a **critical-difference diagram** |
| Paper's single split | **McNemar test** on paired predictions |
| Per cluster | accuracy per held-out cluster, with min–max ranges |
| Seeds | mean ± std across seeds for the deep models |

**Output:** `statistics/` with the notebook, CSV tables and the figures for the paper.

**Note for the paper:** the splits share training clusters, so they aren't independent. State this, and use the corrected test.

---

### Phase 7: Second dataset (optional, high value)

**Goal:** evidence beyond a single measurement campaign.

| Task | Method | Output | Done when |
|---|---|---|---|
| 7.1 Check DICHASUS `cf0x` / `cf1x` | download (DaRUS, CC BY 4.0), read `spec.json`, find how LoS/NLoS is marked, frequency, bandwidth, antennas | `additional_datasets/dichasus/NOTES.md` | data format and labels understood |
| 7.2 Loader and preprocessing | CSI → per-antenna normalised amplitude → same compact input (subcarriers resampled if needed) | `additional_datasets/dichasus/preprocess.ipynb` | per-environment files as for Argos |
| 7.3 Within-dataset evaluation | leave-region-out (or leave-day-out) splits | results files | — |
| 7.4 Cross-dataset test | train on Argos, test on DICHASUS (and the reverse), with the physical and compact features | results table | — |

If DICHASUS turns out to be unusable, use **eWINE UWB** (public, 7 indoor environments) with the physical features only, and state the limitation.

---

### Phase 8: Literature check

**Goal:** a correct related-work section and a defensible novelty claim.

| Task | Search topics |
|---|---|
| 8.1 LOS/NLOS identification with CSI | Wi-Fi CSI, massive MIMO, UWB; ML and DL methods since 2017 |
| 8.2 Cross-environment / domain generalisation in wireless sensing | DANN, Group DRO, IRM, test-time adaptation applied to CSI |
| 8.3 Use of the Argos (TVT) dataset | which papers used it, and for what |
| 8.4 Evaluation practice | single split vs cross-environment protocols in CSI papers |

**Output:** `literature/README.md`, with a table of related works (method, data, evaluation, limitations) and a short statement of what is new here. It replaces the paper's current Table 1.

---

### Phase 9: Manuscript

**Goal:** a full rewrite. The current manuscript contains content from another paper (Algorithm 1, the EAZR conclusion), placeholders and inconsistent numbers (`paper_review_findings/README.md`).

**Structure:**

| Section | Contents | Source |
|---|---|---|
| 1 Introduction | problem, why models fail in new environments, contributions | Phases 4, 8 |
| 2 Related work | updated table | Phase 8 |
| 3 Dataset | Argos campaign, clusters, recording conditions (gain, SNR, spurs) | `dataset_description/` |
| 4 Preprocessing | CSI extraction, per-antenna normalisation, training-only scaling; why averaging raw samples fails | `PIPELINE_README.md` |
| 5 Physical characterisation | CV, K-factor, eigenvalue share per window | Phase 5 |
| 6 Methods | SVM, RF, CNN-LSTM, CNN-Transformer, the new method; epoch selection on held-out environments | Phases 3, 4 |
| 7 Evaluation protocol | paper's split + 80 splits; statistics | Phase 6 |
| 8 Results | main table, per-cluster results, ablations | Phases 1–4, 6 |
| 9 Robustness | spur exclusion, SNR equalisation, seeds, second dataset | Phases 2, 7 |
| 10 Discussion and limitations | one campaign, outdoor only, labels per cluster | — |
| 11 Conclusion | — | — |

**Planned figures:**
1. Measurement map and frame structure (1120 = 5 × 224 slots)
2. Preprocessing pipeline diagram (old vs new)
3. Per-subcarrier amplitude, LOS vs NLOS (`data_exploration/`)
4. Physical feature distributions per cluster (Phase 5)
5. Model architectures (CNN-LSTM, CNN-Transformer, new method)
6. Accuracy over all splits (sorted, paper's split highlighted)
7. Accuracy per held-out cluster for all models
8. Critical-difference diagram (Phase 6)
9. SNR robustness curve (Phase 2)

**Planned tables:**
1. Related work
2. Dataset: clusters, frames, recording conditions
3. Old vs new preprocessing
4. Main results: all models × both protocols, mean ± std [95% CI], p-values, effect sizes
5. Ablation (normalisation, input size, validation method, new-method components)
6. Robustness (spur, SNR, seeds)
7. Cross-dataset results (if Phase 7 is done)

**Checklist before submission:**
- [ ] every number in the text matches a results file
- [ ] all figures generated by notebooks, not edited by hand
- [ ] dataset artefacts stated in Section 3
- [ ] the paper's single-split result not used as the headline
- [ ] limitations stated
- [ ] no content from other papers

---

### Phase 10: Reproducibility and submission

| Task | Detail |
|---|---|
| 10.1 Code package | `preprocessing_v2/` notebooks, core modules, split definitions, `requirements.txt` (TensorFlow environment: `dl_env`, Python 3.11) |
| 10.2 Results package | all `splits_*.jsonl` files and summary tables |
| 10.3 Public repository | e.g. GitHub, linked in the paper, after the co-authors agree |
| 10.4 Journal choice | decided after Phases 4 and 7: mid-tier if they are partial, upper-mid if both succeed |
| 10.5 Cover letter | contribution, data availability, statement on the artefacts |

---

## 4. Order of work and effort

| Order | Phase | Effort (writing) | Run time on this Mac | Depends on |
|---|---|---|---|---|
| 1 | 1 Fair baselines | small | about 2–4 h (SVM/RF, 80 splits each) | — |
| 2 | 2.1 Spur exclusion | small | about 3–5 h (all models) | — |
| 3 | 6 Statistics notebook | medium | minutes | any results |
| 4 | 3 CNN-Transformer | medium | about 2–3 h per seed | — |
| 5 | 2.2–2.4 SNR tests, augmentation, seeds | small–medium | about 4–6 h | 1, 3 |
| 6 | 8 Literature check | medium | — | — |
| 7 | 5 Physical characterisation | medium | about 30 min | — |
| 8 | 4 New method | large | about 6–10 h | 6, 8 |
| 9 | 7 Second dataset | large | depends on data | — |
| 10 | 9 Manuscript | large | — | all |
| 11 | 10 Package and submission | small | — | 9 |

---

## 5. Decision points

| After | Decision |
|---|---|
| Phase 1 | **If SVM/RF on 52 inputs ≈ CNN-LSTM:** the main contribution becomes the representation and the evaluation protocol, and the deep model becomes secondary. **If the CNN-LSTM is clearly better:** keep the deep model as the main method. |
| Phase 2.1 | **If accuracy drops clearly without the spur subcarriers:** use the spur-free input as the default for all later work. |
| Phase 2.2 | **If accuracy drops strongly at equal SNR:** make noise augmentation (2.3) part of every model. |
| Phase 8 | **If domain-invariant learning is already published for this task:** choose a different emphasis for Phase 4 (e.g. physics-informed fusion, delay-domain input, test-time adaptation). |
| Phase 4 | **If the new method doesn't improve the worst splits:** report it as a negative result; the paper still stands on Phases 1–3, 5 and 6. |
| Phase 7 | **If no second dataset is usable:** state "unseen clusters of one campaign" clearly and drop broader generalisation claims. |

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Results rely on recording artefacts (SNR, spur) | Phase 2: report results with and without them |
| Split results aren't independent | corrected statistical test, and state the limitation |
| The new method isn't new | Phase 8 before Phase 4 |
| No second dataset | clear scope statement; physical features as dataset-independent evidence |
| TensorFlow environment problems | always use the `dl_env` kernel; models run one process per split |
| Long run times | `TRAIN_SUBSAMPLE` settings for quick checks; resumable runs |

---

## 7. Progress

| Phase | Status |
|---|---|
| 1 Fair baselines | ☐ not started |
| 2 Robustness | ☐ not started |
| 3 CNN-Transformer | ☐ not started |
| 4 New method | ☐ waiting for Phase 8 |
| 5 Physical characterisation | ☐ not started (cluster-level analysis done) |
| 6 Statistics | ☐ not started |
| 7 Second dataset | ☐ not started |
| 8 Literature check | ☐ not started |
| 9 Manuscript | ☐ not started |
| 10 Submission | ☐ not started |
