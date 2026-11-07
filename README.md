# HushWave

Single-channel speech noise reduction built from scratch in the STFT domain.
Dept. of ECE, NIT Goa, Digital Signal Processing Mini Project 6.

The algorithms (noise PSD tracking, spectral subtraction, Wiener, MMSE-LSA, streaming engine)
are written from first principles in `src/hushwave/`. No deep learning is used.
Every number reported by this project is produced by code and written to
`results/results.json`; none are typed by hand.

## Quick start

```bash
make setup     # create .venv, install dependencies, write requirements.lock
make verify    # lint, type-check, tests with coverage, results-file check
```

## Experiments (M8)

```bash
# put your five recordings in data/recordings (raw WAVs of any rate or channel count are fine):
#   Fan_noise, corridor_noise, Speech_quiet, Speech_in_fan, speech_in_corridor
make data                 # LibriSpeech + DEMAND (needs network), writes the split manifest
make experiments          # tune on DEV, evaluate ONCE on TEST, score real recordings (~20-40 min)
make figures              # results/figures/*.png
make results-doc          # docs/RESULTS.md, generated from results/results.json
make experiments-quick    # 10-minute rehearsal into results/quick (not a result)
```

TEST is evaluated once per frozen setting (`results/tuning.json`); a second run is refused
unless `--force-retest` is passed, and every evaluation is logged in `results.json`.

## Real-time demonstration

```bash
make firmware-check  # compile the firmware for this computer and compare it with the Python engine
make demo-live       # microphone through the ESP32-S3 (Chrome or Edge, headphones)
make examples        # replay clips from data/recordings, written to examples/
make demo-replay     # original, spectral subtraction and HushWave, switchable during playback
```

The browser records the microphone and plays the result. The denoiser itself runs in the firmware on the ESP32-S3
(`firmware/hushwave_esp32/`, flashing steps in `docs/FIRMWARE.md`, serial protocol in `docs/PROTOCOL.md`). The page can also use the
same engine in Python on the computer for comparison. All settings are those of `results/tuning.json`; the firmware constants are
generated from it. See `docs/DEMO.md` for the session script and troubleshooting.

## Report (M10)

```bash
make report          # tables and numbers from results.json, docs/VIVA.md, report/hushwave_report.pdf
```

Every number in the report is a macro generated from `results/results.json`. Before building, the script checks the
qualitative statements of the text against the data and stops when one no longer holds.

## Layout

- `src/hushwave/`: the library
- `firmware/`: ESP32-S3 firmware and its test rig
- `demo/web/`: the demonstration interface
- `scripts/`: command-line tools (verification, data, experiments, demonstration, report)
- `tests/`: pytest suite
- `docs/`: theory, results, traceability, limitations, viva notes, demonstration guide, submission checklist
- `results/`: generated results and figures
- `report/`: the project report (LaTeX source and PDF)
- `examples/`: replay clips

## License

MIT, see `LICENSE`.
