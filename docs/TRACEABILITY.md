# Requirement traceability

Each row maps a requirement of the project brief to the report, the code, the tests, and the command that reproduces it.
Chapter and section numbers refer to `report/hushwave_report.pdf`.

| Requirement | Report | Code | Tests | Reproduce |
|---|---|---|---|---|
| Implement at least one DSP algorithm yourself | Chapters 1 and 3 | `fft.py`, `windows.py`, `stft.py`, `filters.py`, `noise_est.py`, `gains.py`, `enhancer.py` | `test_fft.py`, `test_stft.py`, `test_filters.py`, `test_noise_est.py`, `test_gains.py`, `test_enhancer.py` | `make test` |
| Mathematical formulation | Chapters 1 and 3, `docs/THEORY.md` | `theory.py`, `docfill.py` | `test_theory.py`, `test_docfill.py` | `make docs` |
| Time-domain analysis | Section 5.4, Figure 5.4 | `figures.py` | `test_figures.py` | `make figures` |
| Frequency-domain analysis | Sections 5.4 and 5.5, Figures 5.5 and 5.6 | `figures.py`, `spectral.py` | `test_figures.py`, `test_spectral.py` | `make figures` |
| Raw against processed with numerical measures | Chapter 5, Tables 5.1 and 5.2 | `metrics.py`, `experiments.py`, `aggregate.py`, `stats.py`, `nonintrusive.py` | `test_metrics.py`, `test_experiments.py`, `test_aggregate.py`, `test_stats.py`, `test_nonintrusive.py` | `make experiments`, `make results-doc` |
| Sampling frequency justified | Chapter 2 | `theory.py` (aliasing and quantisation experiments) | `test_theory.py` | `make docs` |
| Microcontroller in the signal path | Chapters 3 and 5.6 | `firmware/hushwave_esp32/`, `mcu.py` | `test_mcu.py`, `test_js.py` | `make firmware-check` |
| Working real-time demonstration | Chapters 3 and 5.6 | `enhancer.py` (streaming), `live.py`, `demo_server.py`, `demo/web/` | `test_live.py`, `test_demo_server.py`, `test_web_assets.py`, `test_examples.py`, `test_js.py` | `make demo-live`, `make demo-replay` |
| Report | all | `reportgen.py`, `reportfigs.py`, `report/report.tex` | `test_reportgen.py`, `test_reportfigs.py` | `make report` |
| Reproducibility | Chapter 4 | `scripts/verify.py`, `requirements.lock` | whole suite | `make verify` |

Rules the project keeps and where they are enforced:

| Rule | Enforced by |
|---|---|
| Algorithms from scratch | code review of `src/hushwave/`; external packages are listed in the report, Chapter 4 |
| No typed numbers | `docfill.render` (theory), `resultsdoc.py` (results), `reportgen.py` (report and viva notes) |
| Tune on development data, evaluate test once | `pipeline.stage_test` refuses a second run; `results.json` records every evaluation |
| Report text matches the data | `reportgen.check_claims` stops `make report` when a statement no longer holds |
| Plain text, no em dashes | `tests/test_text_style.py` |
