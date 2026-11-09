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
2. Open `firmware/hushwave_esp32/hushwave_esp32.ino`. The folder can stay in WSL: use `\\wsl.localhost\Ubuntu\home\kartikey\dsp_mini\hushwave\firmware\hushwave_esp32`.
   If the IDE refuses a network path, copy the folder to `Documents\Arduino\`.
3. Tools menu: Board **ESP32S3 Dev Module**, Upload Speed **921600**, and then the settings for the port the cable is in:

   | Cable in the port marked | USB CDC On Boot | USB Mode | Upload Mode |
   |---|---|---|---|
   | **USB** (native USB, the board shows up as USB Serial Device) | **Enabled** | **Hardware CDC and JTAG** | UART0 / Hardware CDC |
   | **UART** (USB-UART bridge chip) | **Disabled** | any | UART0 / Hardware CDC |

   With the cable in the native USB port and USB CDC On Boot **Disabled**, `Serial` is a hardware UART on pins and nothing answers over USB.
   Upload again after changing a setting.
4. Select the COM port and press Upload. If the upload does not start, hold BOOT, press RST, release BOOT, and upload again.
5. After the upload, press RST once. The board starts the sketch and may appear under a different COM number than the one used for the upload
   (the native USB port is re-created on every reset). Use the number that Device Manager shows now.
6. Close the Arduino IDE and its Serial Monitor. Only one program can hold the port, and the browser needs it.

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

## Testing the board without the browser

`scripts/mcu_probe.py` opens the port, says hello, streams 200 test blocks and prints the board's processing time and the round trip.
It needs the Windows Python and pyserial, because the COM port belongs to Windows:

```
py -m pip install pyserial
copy \\wsl.localhost\Ubuntu\home\kartikey\dsp_mini\hushwave\scripts\mcu_probe.py %USERPROFILE%\
py %USERPROFILE%\mcu_probe.py --list
py %USERPROFILE%\mcu_probe.py --port COM10
```

`--watch 20` prints every change of the port list for 20 seconds; press RST while it runs to see the COM number change.
`PASS` means the board and the firmware work, and a problem in the page lies in the browser or in another program that holds the port.
`FAIL` prints what was seen on the port and what to check.

## Memory and speed

The engine needs about 31 kB of RAM (the ESP32-S3 has 512 kB). One block of 16 ms has to be processed in less than 16 ms; the page shows the
measured time. The firmware is compiled with `-O3` for the engine file when built by the Arduino IDE.
