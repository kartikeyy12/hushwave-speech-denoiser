# Faculty approval request (ready to send)

**To:** the DSP course faculty
**Subject:** DSP Mini Project 6 (Speech Noise Reduction): platform approval, Roll Nos. 20, 21, 22

Respected Sir/Madam,

We are the group for Digital Signal Processing Mini Project 6, Speech Noise Reduction (Roll Nos. 20, 21 and 22, Third-Year B.Tech ECE). The project sheet allows another suitable platform subject to faculty approval, so we are writing to request your approval of the following plan.

**Proposed platform.** A laptop with a microphone (built-in or USB), running our own Python implementation of the processing chain. We write the DSP ourselves: FFT, short-time Fourier transform with overlap-add, a biquad filter, noise power-spectral-density tracking and three gain rules (spectral subtraction, Wiener, MMSE log-spectral amplitude). Library calls are used only for file input/output, plotting, special functions and as clearly labelled baselines for comparison.

**How this meets the minimum deliverables.** We will justify the sampling frequency and document acquisition, implement the algorithms ourselves with the mathematical formulation, present time- and frequency-domain analysis, compare raw and processed signals with numerical measures, and demonstrate a live near-real-time system (microphone to headphones) with a pre-recorded replay as backup.

**Questions we would be grateful for your guidance on:**

1. Is a laptop with a microphone acceptable as the platform, or do you require an embedded board (ESP32, STM32 or a DSP kit) for full marks?
2. How do you define "near-real-time" for the demo? We propose frame-based processing with a total delay of about one frame, and compute time well below the audio duration.
3. May we use public speech and noise datasets (LibriSpeech, DEMAND) together with our own noise recordings for evaluation?
4. May we use open-source packages for baseline comparison and for the PESQ and STOI quality metrics, provided the evaluated algorithms are our own implementations?
5. What are the expected formats and dates for the report, the live demonstration and the viva?

Thank you for your time.

Yours sincerely,
Kartikey Tiwari
Roll No. 20, Third-Year B.Tech, Dept. of ECE, NIT Goa
(on behalf of Roll Nos. 20, 21 and 22)
