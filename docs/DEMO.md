# Demonstration guide

## Before the session

```bash
cd ~/dsp_mini/hushwave
source .venv/bin/activate
make examples          # builds examples/ from data/recordings (once)
make demo-replay       # opens http://localhost:8765/#replay
```

Open the address in **Chrome or Edge on Windows**. WSL2 forwards `localhost` to Windows. If the page does not open,
run `python scripts/demo.py replay --host 0.0.0.0` and use the WSL address shown by `hostname -I`.

For the live mode use `make demo-live`, allow the microphone when the browser asks, and wear headphones.
Firefox cannot open a 16 kHz audio context, so the live mode refuses to start there.

## Three-minute script

1. **Replay, speech in fan noise.** Press Space. Switch with keys 1, 2, 3 between Original, Spectral subtraction, HushWave.
   Say: the three versions share one level, the spectrogram shows the noise floor in the pauses.
2. **Point at the table under the plot.** Say: these are measurements of the whole recording, taken in pauses and in speech blocks,
   because no clean reference exists. The noise reduction is modest, and we report it as it is.
3. **Live.** Start, speak, toggle Enhancement. Say: the browser only captures and plays audio, the signal goes through the same
   streaming engine that was evaluated. The delay shown is block plus algorithmic delay plus round trip plus buffer.
4. **If the microphone fails.** Go back to replay. It needs no audio input.

## If something goes wrong

| Problem | Action |
|---|---|
| Page does not open | `python scripts/demo.py replay --host 0.0.0.0`, open the WSL address |
| Port in use | `python scripts/demo.py replay --port 8800` |
| Live mode says the browser runs audio at another rate | Use Chrome or Edge |
| Microphone denied | Click the lock icon in the address bar, allow the microphone, press Start again |
| Crackling in live mode | Close other audio applications; the interface counts underruns |
| `examples/manifest.json` missing | `make examples` |

Nothing in the demonstration is retuned. The settings are read from `results/tuning.json`.
