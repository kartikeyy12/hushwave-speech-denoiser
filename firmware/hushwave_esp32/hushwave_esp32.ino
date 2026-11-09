// HushWave on the ESP32: real-time speech noise reduction over USB serial.
//
// The computer captures the microphone and sends 16 ms blocks (256 samples, 16 kHz, 16 bit) to the
// board. The board runs the denoiser and sends every block back, and the computer plays it.
// All signal processing happens here, in hushwave_core.cpp.
//
// Board: ESP32-S3 (Arduino IDE: "ESP32S3 Dev Module"). Cable in the native USB port: set
// USB CDC On Boot to Enabled and USB Mode to Hardware CDC and JTAG, so that Serial is the USB port.
// Close the Serial Monitor before using the browser: only one program can hold the port.

#include "hushwave_core.h"
#include "hushwave_protocol.h"

static const unsigned long kBaud = 921600;

static hushwave::Engine engine;
static hushwave::Link device_link(&engine);
static uint8_t rx[512];

static uint32_t clock_us() { return static_cast<uint32_t>(micros()); }

void setup() {
  Serial.setRxBufferSize(4096);
  Serial.begin(kBaud);
}

void loop() {
  int waiting = Serial.available();
  while (waiting > 0) {
    const size_t want = static_cast<size_t>(waiting) > sizeof(rx) ? sizeof(rx) : static_cast<size_t>(waiting);
    const size_t got = Serial.readBytes(rx, want);
    for (size_t i = 0; i < got; ++i) {
      if (device_link.feed(rx[i])) {
        const size_t n = device_link.respond(clock_us);
        if (n > 0) Serial.write(device_link.tx(), n);
      }
    }
    waiting = Serial.available();
  }
}
