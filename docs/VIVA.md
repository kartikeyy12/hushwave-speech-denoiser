# Viva preparation

Generated from `docs/VIVA.template.md` by `make report`. Every number is read from `results/results.json`.
Answers are short on purpose: say the main point, then stop and let the examiner ask.

## The project in one minute

HushWave reduces background noise in single-channel speech. The signal is cut into 32 ms frames with 50 percent overlap,
transformed with a radix-2 FFT, and each time-frequency cell is multiplied by a gain computed from a tracked noise power
estimate. The result is overlap-added back into a signal. The FFT, windows, STFT, filter, trackers, gain rules, metrics
and Welch estimator are written in the project. On a held-out test set of 24 utterances (1320 cases) the mean SNR
improvement is 5.44 dB, against 5.20 dB for tuned spectral subtraction. The advantage is small and consistent.
The system does not improve intelligibility, and on two real recordings its noise reduction in pauses is modest.

## Basics

**Why 16 kHz?** Speech information that matters for intelligibility reaches about 8 kHz, and 16 kHz sampling represents up
to 8 kHz. It matches the corpora and the wide-band PESQ and STOI tools. A higher rate wastes computation on empty bins and
8 kHz would cut the fricatives.

**What is aliasing and did you show it?** Content above half the sampling rate folds back into the band. We decimate a
two-tone signal from 48 kHz to 16 kHz: the 11 kHz tone folds to 5000 Hz at 0.0 dB relative to the in-band
tone without a filter, and at -68.1 dB with a linear-phase low-pass.

**Why 16 bits?** A full-scale sine has about 6.02 N + 1.76 dB SQNR. At 16 bits that is 98.1 dB, and a quiet signal
still has a large margin over any real room noise.

**Why a square-root Hann window and 50 percent overlap?** The same window is used for analysis and synthesis. Its square
sums to one at 50 percent overlap, so an unmodified signal is reconstructed exactly, and the gain is applied between two
smooth windows.

**Why 32 ms?** Speech is close to stationary within a frame of that length, and the frame still resolves the harmonics of
a voice. In the ablation, 16 ms frames are worse and 64 ms frames are not distinguishable from 32 ms.

## The algorithm

**How is the noise estimated when speech is present?** It cannot be measured, so it is tracked. Minimum statistics takes the
minimum of the smoothed periodogram over about 1.5 s and corrects its bias. MCRA compares the smoothed periodogram with its
running minimum, declares speech present above a threshold (3 here), and updates the noise estimate slowly where speech is
likely and quickly where it is not.

**What is the rise limit?** A minimum tracker jumps up when the last pause leaves its window during continuous speech. The
limit lets the estimate fall at once and rise at most 3 dB per second. It was chosen on development data. On the test data it does not give
a significant SNR gain (Holm p = 0.604) and it lowers the noise reduction (10.0 dB with it, 16.6 dB without).
Say so if asked.

**Wiener gain against spectral subtraction?** The Wiener gain is xi / (1 + xi), with xi the a priori SNR. Subtraction
subtracts a noise magnitude and floors the result. On the same pipeline the Wiener gain is better on SNR and has less musical
noise.

**What does decision-directed estimation do?** It estimates the a priori SNR from the previous enhanced frame and the current
a posteriori SNR, with a memory constant (0.9 here). It smooths the gain over time, which suppresses isolated bursts of residual
noise. Against the plain-SNR Wiener gain it adds 0.75 dB median SNR improvement.

**What is musical noise?** Isolated spectral peaks that survive in the pauses and sound like short tones. We measure it as the
kurtosis ratio of the noise-only spectra, output over input. HushWave has a higher ratio (2.56) than spectral subtraction (2.16).
It is one of the weaknesses.

**Why a zero-phase spectral high-pass?** The causal 80 Hz biquad shifts the phase of low harmonics, and against an unfiltered
clean reference that phase shift counts as error. The same magnitude as a per-bin gain has no phase. Replacing it by the
biquad costs 4.4 dB of SNR improvement and removing the high-pass costs 1.7 dB.

**What is the delay?** The streaming engine waits for the last frame that contains a sample, so the delay is the frame length
minus one: 511 samples, 31.9 ms. It is independent of the block size. The streamed output is identical to the batch output.

**Is it real time?** For 16 ms blocks the real-time factor is 0.020 and the slowest block takes 0.56 ms against a budget of
16 ms. That is one laptop CPU, not an embedded target.

## Evaluation

**Why separate development and test sets?** Parameters were tuned on development data (220 cases). The test set was evaluated
1 time. A second evaluation is refused by the code unless forced and would have to be disclosed. Otherwise the result would
describe how well we fitted the test data.

**Why is the utterance the unit of analysis?** Cases built from the same utterance are not independent. Metrics are averaged per utterance, intervals
are bootstrap intervals over utterances, and paired tests are Wilcoxon signed-rank tests with Holm correction.

**What do the baselines show?** B1 and B4 differ from B5 in one component, so they isolate that component. B2 (scipy Wiener filter)
and B3 (noisereduce) are generic tools.

**Why does B3 look bad if it removes the most noise?** It removes 21.3 dB of noise but also the speech: its SNR improvement is -1.93 dB.
Noise reduction alone is not a quality measure.

**Is the improvement over spectral subtraction meaningful?** It is statistically consistent (median 0.26 dB, Holm p <0.001) and small. PESQ is equal
(0.25 against 0.24, Holm p = 0.415). Do not oversell it.

**Why does STOI not improve?** We did not test the cause. A common explanation is that gain-based enhancement removes noise but does
not restore speech components that the noise masked, and it can remove weak speech together with the noise. No method of this
family improves STOI here, and B3 changes it by -0.035.

**Did the development results carry over to the test set?** Yes. The tuning objective of B5 against B1 is 5.06 against 4.65 on the development set and
4.49 against 4.11 on the test set. The margin is stable.

## Weak points (say them before the examiner does)

- Musical-noise ratio of B5 is the highest of the classical systems.
- No benefit on transient noise (-0.2 dB), and spectral subtraction is better on babble (1.4 dB against 1.1 dB).
- Real recordings: noise reduction in pauses is 2.8 dB (fan) and 1.9 dB (corridor), spectral subtraction reaches 3.9 dB and 2.1 dB.
- The minimum-statistics tracker is slightly better than MCRA on the test set in SNR improvement (about 0.1 dB, Holm p = 0.011), although MCRA was chosen on development data.
- The own-voice corpus has too few segments for corrected significance tests.
- Mixtures are synthetic. Real rooms add reverberation.

## What is yours and what is a library?

Written in the project: FFT, windows, STFT and overlap-add, biquad, noise trackers, gain rules, Welch estimator, SNR,
segmental SNR, noise reduction, speech distortion, log-spectral distance, musical-noise ratio, the experiment runner,
bootstrap intervals, the streaming engine and the demo server. External: NumPy and SciPy for arrays and special functions, soundfile for files,
Matplotlib for plots, `pesq` and `pystoi` for the two perceptual metrics, `noisereduce` and `scipy.signal.wiener` as
comparison systems, `scipy.stats.wilcoxon` for the test statistic.

## The demo

Replay mode plays two excerpts of the real recordings in three versions with one level and switches without a gap.
Live mode sends the microphone through the same streaming engine and shows the delay and processing time. The browser is only
the audio device. If the microphone fails, replay mode needs no audio input.

## If asked what you would do with more time

Add a musical-noise term to the tuning objective, a detector that treats transients as noise, longer recordings with more
pauses, and a fixed-point port to a microcontroller.
