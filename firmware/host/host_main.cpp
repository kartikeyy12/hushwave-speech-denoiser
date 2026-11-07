// The firmware on a PC: reads frames on stdin, writes the answers on stdout.
// Used by the tests and by the browser-free checks; it contains no signal processing of its own.
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "hushwave_core.h"
#include "hushwave_protocol.h"

static uint32_t clock_us() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint32_t>(ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000ULL);
}

int main(int argc, char** argv) {
  if (argc > 1 && strcmp(argv[1], "--sizes") == 0) {
    printf("{\"engine_bytes\": %zu, \"link_bytes\": %zu}\n", sizeof(hushwave::Engine), sizeof(hushwave::Link));
    return 0;
  }
  static hushwave::Engine engine;
  static hushwave::Link device_link(&engine);
  uint8_t buffer[4096];
  for (;;) {
    const ssize_t got = read(STDIN_FILENO, buffer, sizeof(buffer));
    if (got <= 0) break;
    for (ssize_t i = 0; i < got; ++i) {
      if (device_link.feed(buffer[i])) {
        const size_t n = device_link.respond(clock_us);
        if (n) {
          fwrite(device_link.tx(), 1, n, stdout);
          fflush(stdout);
        }
      }
    }
  }
  return 0;
}
