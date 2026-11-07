# Submission checklist

Run from the repository root with the environment active.

1. Copy the files of this delivery over the repository, then `pip install -e ".[dev,metrics,audio]"` if packages are missing.
2. `make firmware-params firmware-check` once. Both must finish without an error.
3. Flash the ESP32-S3 as described in `docs/FIRMWARE.md`.
4. `make demo-live` with headphones: select ESP32, Start, speak, switch Enhancement, Record, Stop. Press Save measurements and move
   `mcu_measurements.json` from Downloads into `results/`.
5. `make examples` and `make demo-replay`: play both clips and switch the three versions.
6. `make verify` must end with `OVERALL: PASS`.
7. `make report` (needs LaTeX; the delivery contains a built PDF). The PDF is `report/hushwave_report.pdf`.
8. `git add -A && git commit -m "Real-time ESP32 denoiser, demo, report" && git tag m12 && git push origin main && git push origin m12`.

Do not run `make experiments` again. The test set has been evaluated once, and a second evaluation would have to be disclosed.
