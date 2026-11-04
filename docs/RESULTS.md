# Results (generated)

Generated 2026-11-04T09:21:52+00:00 at commit `591a908` (profile `full`, Python 3.11.18, 16 CPUs). Do not edit by hand: run `make results-doc`.

## 1. Protocol

* Primary corpus: `libri` (24 utterances, 1320 cases). Input SNR grid (dB): -5, 0, 5, 10, 15.
* Hyper-parameters were tuned on DEV only (220 tuning cases; objective: J = mean over DEV tuning cases of (delta SNR + delta segSNR) / 2, dB).
* TEST was evaluated 1 time(s).
* Unit of analysis: utterance: per-utterance means, 95 % bootstrap CI over utterances.

Frozen settings of B5 (`results/tuning.json`):

* rule `wiener`, tracker `mcra`, pre-filter `spectral`, frame 32 ms
* gain parameters: {'dd_alpha': 0.9, 'g_min': 0.03, 'gain_mu': 0.0}
* tracker parameters: {'delta': 3.0, 'rise_db_per_s': 3.0}

DEV objective J per system (dB, higher is better): B1 4.65, B2 -0.29, B3 -2.48, B4 4.40, B5 5.06. Strongest baseline on DEV: **B1**.

## 2. Overall, corpus `libri` (primary)

Mean over utterances with 95 % bootstrap interval. NR: noise reduction on noise-only blocks. SD: speech distortion. MNR: musical-noise ratio (1 = no more peaky than the input).

| system | ΔSNR (dB) | ΔsegSNR (dB) | NR (dB) | SD (dB) | ΔLSD (dB) | MNR | ΔPESQ | ΔSTOI |
|---|---|---|---|---|---|---|---|---|
| **B0** unprocessed | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | n/a | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| **B1** spectral subtraction (G1) | 5.20 [4.58, 5.76] | 3.03 [2.42, 3.60] | 9.48 [8.99, 10.17] | -16.06 [-17.24, -14.81] | -2.89 [-3.36, -2.33] | 2.16 [2.06, 2.26] | 0.24 [0.23, 0.26] | 0.01 [0.00, 0.01] |
| **B2** scipy.signal.wiener | 0.52 [0.32, 0.67] | -0.75 [-1.04, -0.53] | 1.43 [1.39, 1.47] | n/a | -1.33 [-1.81, -0.82] | 1.56 [1.49, 1.64] | -0.04 [-0.07, -0.02] | -0.01 [-0.01, -0.01] |
| **B3** noisereduce (default) | -1.93 [-2.22, -1.67] | -2.46 [-2.80, -2.15] | 21.31 [19.89, 22.78] | n/a | 4.89 [3.29, 6.69] | 5.04 [4.43, 5.71] | -0.21 [-0.25, -0.16] | -0.03 [-0.04, -0.03] |
| **B4** Wiener, plain SNR | 4.74 [4.19, 5.23] | 2.97 [2.35, 3.53] | 7.20 [6.75, 7.84] | -18.99 [-20.47, -17.41] | -2.22 [-2.58, -1.78] | 1.98 [1.89, 2.08] | 0.17 [0.16, 0.18] | 0.01 [0.00, 0.01] |
| **B5** HushWave (full) | 5.44 [4.81, 6.01] | 3.54 [2.89, 4.14] | 10.00 [9.35, 10.97] | -17.25 [-18.56, -15.86] | -2.85 [-3.42, -2.19] | 2.56 [2.39, 2.74] | 0.25 [0.23, 0.27] | 0.00 [0.00, 0.01] |

## 2. Overall, corpus `own` (secondary, single speaker)

Mean over utterances with 95 % bootstrap interval. NR: noise reduction on noise-only blocks. SD: speech distortion. MNR: musical-noise ratio (1 = no more peaky than the input).

| system | ΔSNR (dB) | ΔsegSNR (dB) | NR (dB) | SD (dB) | ΔLSD (dB) | MNR | ΔPESQ | ΔSTOI |
|---|---|---|---|---|---|---|---|---|
| **B0** unprocessed | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | n/a | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| **B1** spectral subtraction (G1) | 5.12 [4.04, 5.83] | 3.17 [1.66, 4.06] | 9.84 [9.05, 10.91] | -16.00 [-17.49, -13.65] | -3.53 [-3.78, -3.22] | 2.12 [1.96, 2.28] | 0.20 [0.15, 0.24] | 0.00 [0.00, 0.01] |
| **B2** scipy.signal.wiener | 0.64 [0.54, 0.76] | -0.57 [-0.78, -0.36] | 1.38 [1.37, 1.39] | n/a | -1.64 [-2.01, -1.32] | 1.49 [1.39, 1.61] | -0.02 [-0.03, -0.01] | -0.01 [-0.01, -0.01] |
| **B3** noisereduce (default) | -2.39 [-2.56, -2.22] | -3.29 [-3.61, -2.93] | 19.48 [18.46, 20.37] | n/a | 2.61 [0.65, 4.33] | 4.43 [3.48, 5.37] | -0.11 [-0.15, -0.06] | -0.04 [-0.05, -0.03] |
| **B4** Wiener, plain SNR | 5.08 [4.33, 5.57] | 3.51 [2.24, 4.28] | 7.53 [6.81, 8.49] | -19.84 [-21.49, -17.20] | -2.73 [-2.98, -2.39] | 1.99 [1.82, 2.15] | 0.15 [0.10, 0.18] | 0.01 [0.00, 0.01] |
| **B5** HushWave (full) | 5.53 [4.34, 6.31] | 3.84 [2.21, 4.74] | 10.49 [9.34, 12.05] | -17.43 [-19.24, -14.61] | -3.58 [-3.88, -3.21] | 2.46 [2.23, 2.69] | 0.22 [0.15, 0.27] | -0.00 [-0.01, 0.00] |

## 3. ΔSNR (dB) by input SNR (`libri`)

| system | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB |
|---|---|---|---|---|---|
| B0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| B1 | 7.37 [7.10, 7.62] | 6.47 [6.13, 6.81] | 5.59 [5.03, 6.11] | 4.21 [3.37, 4.98] | 2.33 [1.16, 3.41] |
| B2 | 1.00 [0.98, 1.02] | 0.96 [0.92, 1.00] | 0.88 [0.76, 0.95] | 0.48 [0.19, 0.67] | -0.70 [-1.24, -0.29] |
| B3 | 6.59 [6.43, 6.75] | 2.71 [2.46, 2.94] | -1.64 [-1.95, -1.35] | -6.26 [-6.61, -5.94] | -11.06 [-11.43, -10.71] |
| B4 | 6.00 [5.77, 6.23] | 5.54 [5.27, 5.80] | 5.13 [4.63, 5.57] | 4.22 [3.43, 4.90] | 2.79 [1.62, 3.84] |
| B5 | 7.43 [7.13, 7.73] | 6.55 [6.20, 6.89] | 5.81 [5.23, 6.33] | 4.56 [3.70, 5.32] | 2.86 [1.64, 3.96] |

## 3. ΔPESQ-WB by input SNR (`libri`)

| system | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB |
|---|---|---|---|---|---|
| B0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| B1 | 0.08 [0.07, 0.10] | 0.15 [0.13, 0.17] | 0.26 [0.24, 0.28] | 0.35 [0.31, 0.39] | 0.38 [0.34, 0.43] |
| B2 | 0.00 [-0.01, 0.01] | 0.01 [-0.00, 0.01] | -0.01 [-0.03, 0.01] | -0.06 [-0.09, -0.03] | -0.15 [-0.21, -0.10] |
| B3 | -0.00 [-0.02, 0.01] | -0.02 [-0.04, 0.00] | -0.12 [-0.16, -0.08] | -0.32 [-0.39, -0.25] | -0.59 [-0.69, -0.49] |
| B4 | 0.05 [0.05, 0.06] | 0.10 [0.09, 0.12] | 0.17 [0.15, 0.19] | 0.24 [0.22, 0.27] | 0.27 [0.23, 0.31] |
| B5 | 0.10 [0.08, 0.11] | 0.17 [0.15, 0.19] | 0.28 [0.25, 0.30] | 0.35 [0.31, 0.39] | 0.37 [0.32, 0.42] |

## 3. ΔSTOI by input SNR (`libri`)

| system | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB |
|---|---|---|---|---|---|
| B0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| B1 | 0.01 [0.00, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.00, 0.01] | 0.00 [0.00, 0.00] |
| B2 | -0.00 [-0.01, -0.00] | -0.01 [-0.01, -0.00] | -0.01 [-0.01, -0.01] | -0.02 [-0.02, -0.01] | -0.02 [-0.02, -0.02] |
| B3 | -0.03 [-0.03, -0.02] | -0.03 [-0.03, -0.02] | -0.03 [-0.04, -0.03] | -0.04 [-0.04, -0.04] | -0.05 [-0.05, -0.04] |
| B4 | 0.01 [0.00, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.00, 0.01] | 0.00 [0.00, 0.00] |
| B5 | 0.00 [-0.00, 0.01] | 0.01 [0.00, 0.01] | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] | 0.00 [-0.00, 0.00] |

## 4. ΔSNR (dB) by noise condition (`libri`)

| system | demand-OHALLWAY | demand-PCAFETER | demand-STRAFFIC | own-corridor_noise | own-fan_noise | syn-am | syn-babble | syn-hum | syn-pink | syn-transient | syn-white |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| B1 | 12.0 [10.5, 13.3] | 4.6 [4.0, 5.0] | 7.6 [6.7, 8.5] | 3.6 [3.1, 4.0] | 5.4 [4.9, 6.0] | 2.9 [2.4, 3.3] | 1.4 [1.0, 1.7] | 8.4 [7.5, 9.4] | 6.5 [5.8, 7.1] | -0.1 [-0.4, 0.1] | 4.9 [4.4, 5.4] |
| B2 | -0.5 [-0.6, -0.4] | -0.3 [-0.4, -0.2] | -0.4 [-0.6, -0.3] | -0.2 [-0.4, -0.1] | -0.2 [-0.4, -0.1] | 3.2 [2.9, 3.4] | -0.0 [-0.2, 0.1] | -0.5 [-0.6, -0.4] | 0.1 [-0.1, 0.2] | 0.5 [0.3, 0.6] | 4.1 [3.7, 4.3] |
| B3 | -1.4 [-1.7, -1.1] | -1.9 [-2.2, -1.6] | -1.5 [-1.8, -1.2] | -2.1 [-2.4, -1.9] | -1.9 [-2.2, -1.6] | -2.1 [-2.3, -1.8] | -2.2 [-2.5, -2.0] | -1.3 [-1.7, -1.0] | -1.8 [-2.1, -1.6] | -3.2 [-3.4, -3.0] | -1.8 [-2.1, -1.6] |
| B4 | 12.5 [11.0, 13.9] | 4.0 [3.6, 4.4] | 7.1 [6.3, 7.9] | 3.0 [2.6, 3.4] | 4.9 [4.4, 5.4] | 1.5 [1.1, 1.8] | 0.6 [0.3, 0.9] | 8.7 [7.8, 9.6] | 6.2 [5.6, 6.8] | -0.2 [-0.5, -0.0] | 3.6 [3.1, 4.0] |
| B5 | 12.6 [11.1, 14.0] | 4.6 [4.1, 5.0] | 8.0 [7.0, 8.9] | 3.9 [3.4, 4.4] | 5.5 [4.9, 6.0] | 3.0 [2.5, 3.5] | 1.1 [0.8, 1.4] | 8.7 [7.7, 9.7] | 7.1 [6.4, 7.7] | -0.2 [-0.4, 0.0] | 5.6 [5.0, 6.1] |

## 5. Ablation (`libri`)

Each row changes one component of B5. Test: B5 minus the variant, Wilcoxon on per-utterance means, Holm-corrected within this table.

| variant | ΔSNR (dB) | ΔsegSNR (dB) | NR (dB) | median B5 - variant (ΔSNR) | p (Holm) |
|---|---|---|---|---|---|
| **B5** (full) | 5.44 [4.81, 6.01] | 3.54 [2.89, 4.14] | 10.00 [9.35, 10.97] |  |  |
| A:rule-subtraction gain rule subtraction | 4.62 [3.95, 5.28] | 2.47 [1.93, 2.99] | 14.53 [13.76, 15.63] | 0.74 | <0.001 |
| A:est-minstats noise tracker minstats | 5.55 [4.90, 6.15] | 3.56 [2.91, 4.15] | 11.67 [11.06, 12.57] | -0.11 | 0.011 |
| A:rise-off no rise limit | 5.55 [4.86, 6.22] | 3.22 [2.67, 3.77] | 16.62 [15.92, 17.42] | -0.17 | 0.604 |
| A:prefilter-none pre-filter none | 3.77 [3.56, 3.98] | 3.18 [2.88, 3.47] | 6.57 [6.00, 7.39] | 2.03 | <0.001 |
| A:prefilter-biquad pre-filter biquad | 1.03 [0.42, 1.60] | -0.69 [-1.13, -0.23] | 9.78 [9.12, 10.73] | 4.29 | <0.001 |
| A:weak-dd weak decision-directed memory | 4.98 [4.40, 5.49] | 3.20 [2.56, 3.79] | 7.77 [7.27, 8.48] | 0.47 | <0.001 |
| A:gmin-low lower gain floor | 5.44 [4.81, 6.01] | 3.54 [2.90, 4.14] | 10.05 [9.37, 11.07] | -0.00 | 1.000 |
| A:gmin-high higher gain floor | 5.43 [4.81, 6.00] | 3.50 [2.86, 4.10] | 9.73 [9.17, 10.51] | 0.01 | 0.003 |
| A:frame-16ms frame length 16 ms | 5.02 [4.41, 5.60] | 3.19 [2.59, 3.75] | 11.45 [10.61, 12.63] | 0.41 | <0.001 |
| A:frame-64ms frame length 64 ms | 5.48 [4.85, 6.04] | 3.54 [2.89, 4.15] | 9.27 [8.75, 10.00] | -0.02 | 1.000 |

## 6. B5 against each baseline (`libri`)

Wilcoxon signed-rank on per-utterance means (primary), Holm-corrected over all tests of this table; the case-level p is optimistic because cases share utterances.

| vs | metric | direction | n utts | median B5 - vs | p | p (Holm) | p (cases) |
|---|---|---|---|---|---|---|---|
| B1 | delta_snr_db | higher is better | 24 | 0.26 | <0.001 | <0.001 | <0.001 |
| B1 | delta_segsnr_db | higher is better | 24 | 0.55 | <0.001 | <0.001 | <0.001 |
| B1 | noise_reduction_db | higher is better | 23 | 0.41 | <0.001 | <0.001 | <0.001 |
| B1 | delta_lsd_db | lower is better | 24 | -0.04 | 0.584 | 0.584 | <0.001 |
| B1 | delta_pesq | higher is better | 24 | 0.01 | 0.208 | 0.415 | <0.001 |
| B1 | delta_stoi | higher is better | 24 | -0.00 | <0.001 | <0.001 | <0.001 |
| B2 | delta_snr_db | higher is better | 24 | 5.68 | <0.001 | <0.001 | <0.001 |
| B2 | delta_segsnr_db | higher is better | 24 | 5.26 | <0.001 | <0.001 | <0.001 |
| B2 | noise_reduction_db | higher is better | 23 | 8.21 | <0.001 | <0.001 | <0.001 |
| B2 | delta_lsd_db | lower is better | 24 | -1.39 | <0.001 | <0.001 | <0.001 |
| B2 | delta_pesq | higher is better | 24 | 0.28 | <0.001 | <0.001 | <0.001 |
| B2 | delta_stoi | higher is better | 24 | 0.01 | <0.001 | <0.001 | <0.001 |
| B3 | delta_snr_db | higher is better | 24 | 7.60 | <0.001 | <0.001 | <0.001 |
| B3 | delta_segsnr_db | higher is better | 24 | 6.70 | <0.001 | <0.001 | <0.001 |
| B3 | noise_reduction_db | higher is better | 23 | -10.70 | <0.001 | <0.001 | <0.001 |
| B3 | delta_lsd_db | lower is better | 24 | -7.25 | <0.001 | <0.001 | <0.001 |
| B3 | delta_pesq | higher is better | 24 | 0.47 | <0.001 | <0.001 | <0.001 |
| B3 | delta_stoi | higher is better | 24 | 0.04 | <0.001 | <0.001 | <0.001 |
| B4 | delta_snr_db | higher is better | 24 | 0.75 | <0.001 | <0.001 | <0.001 |
| B4 | delta_segsnr_db | higher is better | 24 | 0.57 | <0.001 | <0.001 | <0.001 |
| B4 | noise_reduction_db | higher is better | 23 | 2.65 | <0.001 | <0.001 | <0.001 |
| B4 | delta_lsd_db | lower is better | 24 | -0.77 | <0.001 | <0.001 | <0.001 |
| B4 | delta_pesq | higher is better | 24 | 0.08 | <0.001 | <0.001 | <0.001 |
| B4 | delta_stoi | higher is better | 24 | -0.00 | <0.001 | <0.001 | <0.001 |

Against the strongest DEV baseline (B1), `libri`:

* ΔSNR (dB): B5 5.44 vs B1 5.20; median paired difference 0.26, Holm-corrected p <0.001: B5 is better.
* ΔsegSNR (dB): B5 3.54 vs B1 3.03; median paired difference 0.55, Holm-corrected p <0.001: B5 is better.
* NR (dB): B5 10.00 vs B1 9.48; median paired difference 0.41, Holm-corrected p <0.001: B5 is better.
* ΔLSD (dB): B5 -2.85 vs B1 -2.89; median paired difference -0.04, Holm-corrected p 0.584: no significant difference.
* ΔPESQ: B5 0.25 vs B1 0.24; median paired difference 0.01, Holm-corrected p 0.415: no significant difference.
* ΔSTOI: B5 0.00 vs B1 0.01; median paired difference -0.00, Holm-corrected p <0.001: B5 is worse.

## 7. Real recordings without a clean reference

no clean reference exists: pauses are found with the matching noise-only recording as yardstick; see THEORY 8.3

### speech_in_fan

64.8 s; noise floor -36.8 dB; pause blocks 1471 of 4050, speech blocks 1202.

| system | pause NR (dB) | pause level (dBFS) | speech level change (dB) | MNR |
|---|---|---|---|---|
| B0 | 0.0 | -42.2 | 0.0 | 1.00 |
| B1 | 3.9 | -46.1 | -0.6 | 2.77 |
| B2 | 1.5 | -43.6 | -0.2 | 2.19 |
| B3 | 15.1 | -57.3 | -9.8 | 6.85 |
| B4 | 2.3 | -44.5 | -0.1 | 2.11 |
| B5 | 2.8 | -45.0 | -0.2 | 2.87 |

### speech_in_corridor

65.2 s; noise floor -34.8 dB; pause blocks 740 of 4077, speech blocks 2510.

| system | pause NR (dB) | pause level (dBFS) | speech level change (dB) | MNR |
|---|---|---|---|---|
| B0 | 0.0 | -37.6 | 0.0 | 1.00 |
| B1 | 2.1 | -39.7 | -0.3 | 1.48 |
| B2 | 0.7 | -38.3 | -0.2 | 0.99 |
| B3 | 20.1 | -57.7 | -11.3 | 6.39 |
| B4 | 1.8 | -39.4 | -0.1 | 1.33 |
| B5 | 1.9 | -39.5 | -0.2 | 1.42 |

## 8. Data-quality flags

* External-metric failures: 0
* Speech-active fraction, `libri`: mean 0.83, range [0.61, 1.00] over 24 utterances
* Speech-active fraction, `own`: mean 0.83, range [0.78, 0.89] over 6 utterances

