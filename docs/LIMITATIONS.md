# Limitations

These are reported in the report (Section 7) and repeated here so that nothing is hidden in a long document.

## What the results support

- HushWave has the best waveform measures (SNR and segmental SNR improvement) of the compared systems at every input SNR.
- The margin over a tuned spectral subtraction is small. It is statistically consistent on LibriSpeech.
- PESQ-WB is equal to spectral subtraction. No gain-based method improves STOI.

## Weaknesses

1. The musical-noise ratio of HushWave is the highest of the classical systems. The tuning objective cannot see musical noise, and the development data switched gain smoothing off.
2. There is no improvement on transient noise, and spectral subtraction is better on babble.
3. On the two real recordings the noise reduction in pauses is no better than that of spectral subtraction.
4. The rise limit and the MCRA tracker were selected on development data. On the test data the rise limit gives no significant SNR gain and costs noise reduction, and the minimum-statistics tracker is slightly better in SNR improvement. Neither was changed after the test evaluation.
5. The own-voice corpus has six test segments, too few for corrected significance tests.

## Threats to validity

- Test mixtures are synthetic. Real rooms add reverberation and non-stationary noise.
- The fan and corridor recordings differ in format (the corridor files are 48 kHz stereo, the others 44.1 kHz dual mono). Digital mixing makes this harmless for mixtures, but it deviates from the recording protocol.
- The tuning objective is a waveform measure. A perceptual objective could select different parameters.
- PESQ and STOI come from external packages and were used as published.
- Timing was measured on one machine and does not transfer to an embedded target.

## Demonstration limits

- The live mode needs Chrome or Edge, a microphone, and headphones. Firefox cannot open a 16 kHz audio context.
- The estimated delay in the live interface leaves out the latency of the audio device.
- The replay clips are excerpts of two recordings and are not a measurement. The numbers beside them are measurements of the whole recordings.
