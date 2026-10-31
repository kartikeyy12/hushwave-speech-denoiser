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

## Layout

- `src/hushwave/`: the library
- `scripts/`: command-line tools (verification, data, experiments, demos)
- `tests/`: pytest suite
- `docs/`: theory, traceability, limitations
- `results/`: generated results and figures
- `report/`: the project report

## License

MIT, see `LICENSE`.
