# Requirement traceability

Each row maps a requirement of the project brief to the report, the code, the tests, and the command that reproduces it.
Section numbers refer to `report/hushwave_report.pdf`.

| Requirement | Report | Code | Tests | Reproduce |
|---|---|---|---|---|
| Implement at least one DSP algorithm yourself | Section 3 | `fft.py`, `windows.py`, `stft.py`, `filters.py`, `noise_est.py`, `gains.py`, `enhancer.py` | `test_fft.py`, `test_stft.py`, `test_filters.py`, `test_noise_est.py`, `test_gains.py`, `test_enhancer.py` | `make test` |
| Mathematical formulation | Sections 2 and 3, `docs/THEORY.md` | `theory.py`, `docfill.py` | `test_theory.py`, `test_docfill.py` | `make docs` |
| Time-domain analysis | Section 5.4, Figure 6 | `figures.py` | `test_figures.py` | `make figures` |
| Frequency-domain analysis | Section 5.4 and 5.5, Figures 7 to 9 | `figures.py`, `spectral.py` | `test_figures.py`, `test_spectral.py` | `make figures` |
| Raw against processed with numerical measures | Section 5, Tables 3 to 9 | `metrics.py`, `experiments.py`, `aggregate.py`, `stats.py`, `nonintrusive.py` | `test_metrics.py`, `test_experiments.py`, `test_aggregate.py`, `test_stats.py`, `test_nonintrusive.py` | `make experiments`, `make results-doc` |
| Sampling frequency justified | Section 2.2 | `theory.py` (aliasing and quantisation experiments) | `test_theory.py` | `make docs` |
| Working real-time or near-real-time demonstration | Sections 5.8 and 6 | `enhancer.py` (streaming), `live.py`, `demo_server.py`, `demo/web/` | `test_live.py`, `test_demo_server.py`, `test_web_assets.py`, `test_examples.py` | `make demo-replay`, `make demo-live` |
| Report | all | `reportgen.py`, `report/report.tex` | `test_reportgen.py` | `make report` |
| Reproducibility | Appendix A | `scripts/verify.py`, `requirements.lock` | whole suite | `make verify` |

Rules the project keeps and where they are enforced:

| Rule | Enforced by |
|---|---|
| Algorithms from scratch | code review of `src/hushwave/`; external packages are listed in the report, Section 1.3 |
| No typed numbers | `docfill.render` (theory), `resultsdoc.py` (results), `reportgen.py` (report and viva notes) |
| Tune on development data, evaluate test once | `pipeline.stage_test` refuses a second run; `results.json` records every evaluation |
| Report text matches the data | `reportgen.check_claims` stops `make report` when a statement no longer holds |
| Plain text, no em dashes | `tests/test_text_style.py` |
