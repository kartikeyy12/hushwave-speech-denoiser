# Recording protocol (do this today)

Goal: five short files that let us test on noise nobody else has. Total time: about 15 minutes.

## What to record

| File name | What | Length | Used for |
|---|---|---|---|
| `fan_noise.wav` | Fan only, nobody speaking | 60 s | noise for synthetic mixing (intrusive metrics) |
| `corridor_noise.wav` | Corridor only, nobody speaking | 60 s | noise for synthetic mixing |
| `speech_quiet.wav` | You reading any English paragraph aloud at normal pace, in the quietest room you have | 40 s | clean reference voice |
| `speech_in_fan.wav` | You reading aloud with the fan running | 30 s | real noisy demo, non-intrusive metrics only |
| `speech_in_corridor.wav` | You reading aloud in the corridor | 30 s | real noisy demo, non-intrusive metrics only |

Only your own voice. Do not record classmates or teachers without asking them.

## Rules that decide whether the data is usable

1. **Same device, same settings, same distance for all five files.** If your teammate's recording mic is better, use it for everything.
2. **Turn off all automatic processing.** No AGC (auto gain), no noise suppression or "voice isolation", no echo cancellation, no "enhance recording". These would hide the noise we are trying to remove. On a phone, use a plain voice-recorder app and a wired or built-in mic, not Bluetooth earbuds.
3. **Distance.** Mouth to mic about 30 cm for speech. For noise-only files, put the mic exactly where it was for speech. Fan about 1 m away on a medium setting.
4. **Level.** Peaks should land between about -20 and -6 dBFS. Our checker warns if the recording is clipped (too loud) or very quiet.
5. **Noise-only files must contain no speech.** Footsteps and distant doors in the corridor are fine and realistic.
6. **Keep the room still**: no music, no phone notifications, no typing during recording.
7. **Record in WAV if the app allows it.** Otherwise M4A or MP3 is fine; the conversion below handles it.

## Getting the files into WSL

Put the five files in your Windows Downloads folder in a folder called `recordings`, then run in WSL (adjust the extension if your phone saves `.m4a`; the loop handles `.wav`, `.m4a` and `.mp3`):

```bash
cd ~/dsp_mini/hushwave
source .venv/bin/activate
mkdir -p data/recordings_raw data/recordings
cp /mnt/c/Users/karti/Downloads/recordings/* data/recordings_raw/
for f in data/recordings_raw/*; do
  name=$(basename "${f%.*}")
  ffmpeg -y -loglevel error -i "$f" -ac 1 -ar 16000 -sample_fmt s16 "data/recordings_raw/${name}_16k.wav"
done
python scripts/prepare_recordings.py data/recordings_raw/*_16k.wav
```

`prepare_recordings.py` prints the level of each file and any warnings, and saves checked 16 kHz mono 16-bit WAVs in `data/recordings/` (names end in `_16k`; rename them to the five names above if you like). Paste its output here. If a warning appears, record that file again.

## How the files are used

- The two noise-only files are mixed with LibriSpeech and with your own voice at SNRs of -5 to 15 dB, exactly like DEMAND noise, with the first half of each file used on DEV and the second half on TEST. These results can use intrusive metrics.
- The two speech-in-noise files have no clean reference, so they get non-intrusive treatment only (noise reduction on noise-only segments, spectrograms, listening). They are reported separately and never mixed into the intrusive tables.
