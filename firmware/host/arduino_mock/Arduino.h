// Minimal stand-in for Arduino.h, only to check that the sketch compiles on a PC.
#pragma once
#include <stddef.h>
#include <stdint.h>

inline unsigned long micros() { return 0; }

struct MockSerial {
  size_t setRxBufferSize(size_t) { return 0; }
  void begin(unsigned long) {}
  int available() { return 0; }
  size_t readBytes(uint8_t*, size_t) { return 0; }
  size_t write(const uint8_t*, size_t n) { return n; }
};
static MockSerial Serial;
