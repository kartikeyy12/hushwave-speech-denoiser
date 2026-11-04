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

## Layout

- `src/hushwave/`: the library
- `scripts/`: command-line tools (verification, data, experiments, demos)
- `tests/`: pytest suite
- `docs/`: theory, traceability, limitations
- `results/`: generated results and figures
- `report/`: the project report

## License

MIT, see `LICENSE`.
