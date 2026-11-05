# Submission checklist

Run from the repository root with the environment active.

1. `git pull` or copy the files of this delivery over the repository, then `pip install -e ".[dev,metrics,audio]"` if packages are missing.
2. `make examples` and check that `examples/manifest.json` and the twelve files next to it exist.
3. `make demo-replay`: play both clips, switch the three versions, seek by clicking the spectrogram.
4. `make demo-live` with headphones: Start, speak, switch Enhancement, Stop.
5. `make report`. The PDF is `report/hushwave_report.pdf`. If `pdflatex` is missing:
   `sudo apt install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended`.
   The delivery already contains a built PDF, so this step can be skipped for the submission itself.
6. Fill in the names in `report/authors.tex` (rows for Roll No. 21 and 22) and in `CONTRIBUTIONS.md`, then run `make report` again.
7. `make verify` must end with `OVERALL: PASS`.
8. `git add -A && git commit -m "M9 to M12: demo, report, documentation" && git tag m12 && git push origin main && git push origin m12`.
9. Print or upload `report/hushwave_report.pdf`. Take the laptop, headphones, and `docs/VIVA.md` to the demonstration.

Do not run `make experiments` again. The test set has been evaluated once, and a second evaluation would have to be disclosed.
