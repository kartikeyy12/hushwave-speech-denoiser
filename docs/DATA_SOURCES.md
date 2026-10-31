# Data sources, licenses and attribution

Nothing in `data/` is committed or redistributed (it is gitignored). `make data` downloads from the publishers, verifies checksums and writes `results/split_manifest.json` and `results/data_checksums.json`. Run `make data-check` first to confirm every URL still resolves.

## Clean speech: LibriSpeech dev-clean

| Item | Value |
|---|---|
| Publisher page | https://www.openslr.org/12 (identifier SLR12) |
| License | CC BY 4.0, as stated on the OpenSLR page |
| Archive | `dev-clean.tar.gz`, listed as 337 MB |
| Checksum | read from the publisher's `md5sum.txt` at download time; a mismatch aborts |
| What we use | 40 utterances of 3 to 8 s: 4 speakers (2 F, 2 M) for DEV and 6 speakers (3 F, 3 M) for TEST, no speaker in both |
| Citation | V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "Librispeech: an ASR corpus based on public domain audio books," ICASSP 2015 (verify before submission) |

## Real noise: DEMAND

| Item | Value |
|---|---|
| Publisher record | https://zenodo.org/records/1227121 (DOI 10.5281/zenodo.1227121) |
| Files used | `STRAFFIC_16k.zip`, `PCAFETER_16k.zip`, `OHALLWAY_16k.zip`; channel 1 (`ch01.wav`) only |
| Checksums | MD5 values pinned in `src/hushwave/datasetup.py`, copied from the record's file list; a mismatch aborts |
| Time split | first half of each 5-minute recording for DEV, second half for TEST |
| Citation | J. Thiemann, N. Ito, E. Vincent, "The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND)," Proc. Meetings on Acoustics, 2013 (verify before submission) |

**License note.** The record is inconsistent. Its metadata block lists Creative Commons Attribution 4.0, while its description and the accompanying `DEMAND.pdf` state Creative Commons Attribution-ShareAlike 3.0 Unported. We treat the stricter text as binding: we attribute the authors and we do not redistribute the audio, which also satisfies the share-alike condition for our own code, since only derived numbers and figures are published. Cite the record's DOI in the report.

## Our own recordings

Recorded by the group under `docs/RECORDING_PROTOCOL.md`, only of our own voice and ambient noise. They stay in `data/recordings/` and are not committed.
