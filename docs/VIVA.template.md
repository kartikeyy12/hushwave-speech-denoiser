# Viva preparation

Generated from `docs/VIVA.template.md` by `make report`. Every number is read from `results/results.json`.
Answers are short on purpose: say the main point, then stop and let the examiner ask.

## The project in one minute

HushWave reduces background noise in single-channel speech. The signal is cut into 32 ms frames with 50 percent overlap,
transformed with a radix-2 FFT, and each time-frequency cell is multiplied by a gain computed from a tracked noise power
estimate. The result is overlap-added back into a signal. The FFT, windows, STFT, filter, trackers, gain rules, metrics
and Welch estimator are written in the project. On a held-out test set of @@NUtt@@ utterances (@@NCases@@ cases) the mean SNR
improvement is @@DsnrBfive@@ dB, against @@DsnrBone@@ dB for tuned spectral subtraction. The advantage is small and consistent.
The system does not improve intelligibility, and on two real recordings its noise reduction in pauses is modest.

## Basics

**Why 16 kHz?** Speech information that matters for intelligibility reaches about 8 kHz, and 16 kHz sampling represents up
to 8 kHz. It matches the corpora and the wide-band PESQ and STOI tools. A higher rate wastes computation on empty bins and
8 kHz would cut the fricatives.

**What is aliasing and did you show it?** Content above half the sampling rate folds back into the band. We decimate a
two-tone signal from 48 kHz to 16 kHz: the 11 kHz tone folds to @@AliasHz@@ Hz at @@AliasNaive@@ dB relative to the in-band
tone without a filter, and at @@AliasFiltered@@ dB with a linear-phase low-pass.

**Why 16 bits?** A full-scale sine has about 6.02 N + 1.76 dB SQNR. At 16 bits that is @@SqnrSixteen@@ dB, and a quiet signal
still has a large margin over any real room noise.

**Why a square-root Hann window and 50 percent overlap?** The same window is used for analysis and synthesis. Its square
sums to one at 50 percent overlap, so an unmodified signal is reconstructed exactly, and the gain is applied between two
smooth windows.

**Why 32 ms?** Speech is close to stationary within a frame of that length, and the frame still resolves the harmonics of
a voice. In the ablation, 16 ms frames are worse and 64 ms frames are not distinguishable from 32 ms.

## The algorithm

**How is the noise estimated when speech is present?** It cannot be measured, so it is tracked. Minimum statistics takes the
minimum of the smoothed periodogram over about 1.5 s and corrects its bias. MCRA compares the smoothed periodogram with its
running minimum, declares speech present above a threshold (@@McraDelta@@ here), and updates the noise estimate slowly where speech is
likely and quickly where it is not.

**What is the rise limit?** A minimum tracker jumps up when the last pause leaves its window during continuous speech. The
limit lets the estimate fall at once and rise at most @@RiseLimit@@ dB per second. It was chosen on development data. On the test data it does not give
a significant SNR gain (Holm p = @@PRiseOff@@) and it lowers the noise reduction (@@NrBfive@@ dB with it, @@RiseOffNr@@ dB without).
Say so if asked.

**Wiener gain against spectral subtraction?** The Wiener gain is xi / (1 + xi), with xi the a priori SNR. Subtraction
subtracts a noise magnitude and floors the result. On the same pipeline the Wiener gain is better on SNR and has less musical
noise.

**What does decision-directed estimation do?** It estimates the a priori SNR from the previous enhanced frame and the current
a posteriori SNR, with a memory constant (@@DdAlpha@@ here). It smooths the gain over time, which suppresses isolated bursts of residual
noise. Against the plain-SNR Wiener gain it adds @@MedDsnrFour@@ dB median SNR improvement.

**What is musical noise?** Isolated spectral peaks that survive in the pauses and sound like short tones. We measure it as the
kurtosis ratio of the noise-only spectra, output over input. HushWave has a higher ratio (@@MnrBfive@@) than spectral subtraction (@@MnrBone@@).
It is one of the weaknesses.

**Why a zero-phase spectral high-pass?** The causal 80 Hz biquad shifts the phase of low harmonics, and against an unfiltered
clean reference that phase shift counts as error. The same magnitude as a per-bin gain has no phase. Replacing it by the
biquad costs @@BiquadLoss@@ dB of SNR improvement and removing the high-pass costs @@NoneLoss@@ dB.

**What is the delay?** The streaming engine waits for the last frame that contains a sample, so the delay is the frame length
minus one: @@DelaySamples@@ samples, @@DelayMs@@ ms. It is independent of the block size. The streamed output is identical to the batch output.

**Is it real time?** For 16 ms blocks the real-time factor is @@Rtf@@ and the slowest block takes @@WorstBlock@@ ms against a budget of
@@BlockBudget@@ ms. That is one laptop CPU, not an embedded target.

## Evaluation

**Why separate development and test sets?** Parameters were tuned on development data (@@NTune@@ cases). The test set was evaluated
@@TestEvals@@ time. A second evaluation is refused by the code unless forced and would have to be disclosed. Otherwise the result would
describe how well we fitted the test data.

**Why is the utterance the unit of analysis?** Cases built from the same utterance are not independent. Metrics are averaged per utterance, intervals
are bootstrap intervals over utterances, and paired tests are Wilcoxon signed-rank tests with Holm correction.

**What do the baselines show?** B1 and B4 differ from B5 in one component, so they isolate that component. B2 (scipy Wiener filter)
and B3 (noisereduce) are generic tools.

**Why does B3 look bad if it removes the most noise?** It removes @@NrBthree@@ dB of noise but also the speech: its SNR improvement is @@DsnrBthree@@ dB.
Noise reduction alone is not a quality measure.

**Is the improvement over spectral subtraction meaningful?** It is statistically consistent (median @@MedDsnrOne@@ dB, Holm p @@PDsnrOne@@) and small. PESQ is equal
(@@PesqBfive@@ against @@PesqBone@@, Holm p = @@PPesqOne@@). Do not oversell it.

**Why does STOI not improve?** We did not test the cause. A common explanation is that gain-based enhancement removes noise but does
not restore speech components that the noise masked, and it can remove weak speech together with the noise. No method of this
family improves STOI here, and B3 changes it by @@StoiBthree@@.

**Did the development results carry over to the test set?** Yes. The tuning objective of B5 against B1 is @@JDevFive@@ against @@JDevOne@@ on the development set and
@@JTestFive@@ against @@JTestOne@@ on the test set. The margin is stable.

## Weak points (say them before the examiner does)

- Musical-noise ratio of B5 is the highest of the classical systems.
- No benefit on transient noise (@@TransientFive@@ dB), and spectral subtraction is better on babble (@@BabbleOne@@ dB against @@BabbleFive@@ dB).
- Real recordings: noise reduction in pauses is @@FanNrFive@@ dB (fan) and @@CorNrFive@@ dB (corridor), spectral subtraction reaches @@FanNrOne@@ dB and @@CorNrOne@@ dB.
- The minimum-statistics tracker is slightly better than MCRA on the test set in SNR improvement (about @@MinstatsGain@@ dB, Holm p = @@PMinstats@@), although MCRA was chosen on development data.
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
