# Firmware for the ESP32-S3

The firmware in `firmware/hushwave_esp32/` runs the tuned denoiser on the board. The computer only records, transports and plays audio.

| File | Content |
|---|---|
| `hushwave_esp32.ino` | Arduino sketch: reads serial bytes, answers frames |
| `hushwave_core.h/.cpp` | The enhancer in single precision: FFT, MCRA with rise limit, Wiener gain, high-pass, overlap-add |
| `hushwave_protocol.h/.cpp` | Frame parser and answers |
| `hushwave_params.h` | Constants, generated from `results/tuning.json` by `make firmware-params`. Do not edit |

The core and the protocol are plain C++. They are compiled for a computer by `make firmware-check`, which streams test signals through the
serial protocol and compares the output with the Python engine. The sketch only adds `Serial` and `micros()`.

## Flashing (Windows, Arduino IDE 2)

1. Install the Arduino IDE and, in Boards Manager, the package **esp32** by Espressif Systems.
2. Open `firmware/hushwave_esp32/hushwave_esp32.ino`. The folder can stay in WSL: use `\\wsl.localhost\Ubuntu\home\karti\dsp_mini\hushwave\firmware\hushwave_esp32`.
   If the IDE refuses a network path, copy the folder to `Documents\Arduino\`.
3. Tools menu: Board **ESP32S3 Dev Module**, Upload Speed **921600**.
   - Cable in the port marked **UART** (USB-UART bridge): USB CDC On Boot **Disabled**.
   - Cable in the port marked **USB** (native USB): USB CDC On Boot **Enabled**.
4. Select the COM port and press Upload. If the upload does not start, hold BOOT, press RST, release BOOT, and upload again.
5. Close the Serial Monitor. Only one program can hold the port, and the browser needs it.

Compile once before the demonstration day and read the messages. The processing core and the protocol were built and tested on a computer
with warnings treated as errors; the sketch was checked against a stand-in for the Arduino header.

## Using it

```bash
make demo-live        # WSL: starts the page at http://localhost:8765/#live
```

Open the address in Chrome or Edge on Windows, keep **ESP32** selected, press Start, choose the board's COM port, and allow the microphone.
The page shows the processing time per block reported by the board.

After a session, **Save measurements** downloads `mcu_measurements.json`. Put it in `results/` and run `make report`: the report then states the
measured processing time on the board.

## Memory and speed

The engine needs about 31 kB of RAM (the ESP32-S3 has 512 kB). One block of 16 ms has to be processed in less than 16 ms; the page shows the
measured time. The firmware is compiled with `-O3` for the engine file when built by the Arduino IDE.
