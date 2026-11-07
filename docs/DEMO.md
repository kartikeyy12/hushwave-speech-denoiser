# Demonstration guide

## Setup

```bash
cd ~/dsp_mini/hushwave
source .venv/bin/activate
make examples          # replay clips from data/recordings (once)
make demo-live         # http://localhost:8765/#live
```

Open the address in **Chrome or Edge on Windows**. WSL2 forwards `localhost` to Windows. If the page does not open, run
`python scripts/demo.py live --host 0.0.0.0` and use the WSL address shown by `hostname -I`.
The board is flashed once (`docs/FIRMWARE.md`). Plug it in before you open the page and close the Arduino Serial Monitor.

## Live mode with the ESP32 (about three minutes)

1. Keep **ESP32** selected and press **Start**. Choose the board's COM port in the browser dialog and allow the microphone. Wear headphones.
2. Speak, and switch **Enhancement** on and off. Say: the microphone is captured by the browser, every block of 16 ms goes over USB to the
   ESP32, the firmware denoises it, and the block comes back and plays. The browser does no processing.
3. Point at the numbers: processing time per block (measured by the board), round trip, algorithmic delay, estimated total delay.
4. Press **Record**, speak for ten seconds, press **Stop recording**. Two players appear, the original and the denoised signal, aligned in time.
   They can be downloaded as WAV files.
5. Stop, then **Save measurements**. Move `mcu_measurements.json` to `results/` so that the next `make report` states the board timing.

**This computer** runs the same engine in Python on the laptop. Use it to show that both give the same audio, or as a fallback if the board is not available.

## Replay mode

`make demo-replay` plays two excerpts of the real recordings in three versions (original, spectral subtraction, HushWave) at one level.
Space plays and pauses, keys 1 to 3 switch the version, clicking the spectrogram seeks. The table below the plot holds the measurements of the whole recording.
Replay needs no microphone and no board.

## If something goes wrong

| Problem | Action |
|---|---|
| Page does not open | `python scripts/demo.py live --host 0.0.0.0`, open the WSL address |
| Port 8765 in use | `python scripts/demo.py live --port 8800` |
| ESP32 button greyed out | The browser has no Web Serial. Use Chrome or Edge on a computer |
| "The board did not answer" | Close the Serial Monitor and any other program that holds the port, press RST on the board, press Start again. Check that the sketch is flashed |
| No COM port in the list | Try another cable (some cables carry power only) and the other USB port of the board. Install the driver of the USB-UART chip (CP210x or CH340) |
| Microphone denied | Click the lock icon in the address bar, allow the microphone, press Start again |
| Browser runs audio at another rate | Use Chrome or Edge |
| Crackling | Close other audio programs. The page counts underruns, dropped and lost blocks |
| `examples/manifest.json` missing | `make examples` |

Nothing in the demonstration is retuned. The settings are read from `results/tuning.json`, and the firmware constants are generated from the same file.
