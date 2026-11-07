// Frame protocol between the browser (or any host) and the board. See docs/PROTOCOL.md.
//
//   A5 5A | type | seq | len (u16 LE) | payload | crc16 (LE)
//
// crc16 is CRC-16/CCITT-FALSE over type, seq, len and payload.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "hushwave_core.h"

namespace hushwave {

constexpr uint8_t kSync0 = 0xA5;
constexpr uint8_t kSync1 = 0x5A;
constexpr uint8_t kTypeAudioIn = 0x01;
constexpr uint8_t kTypeAudioOut = 0x81;
constexpr uint8_t kTypeHello = 0x02;
constexpr uint8_t kTypeHelloReply = 0x82;
constexpr uint8_t kTypeReset = 0x03;
constexpr uint8_t kTypeResetAck = 0x83;
constexpr uint8_t kProtocolVersion = 1;
constexpr size_t kMaxPayload = 1024;
constexpr size_t kAudioBytes = kHop * 2;
constexpr size_t kMaxFrame = 2 + 4 + kMaxPayload + 2;

uint16_t crc16(const uint8_t* data, size_t length);

class Link {
 public:
  explicit Link(Engine* engine);
  // Feed one received byte. Returns true when a complete, valid frame is waiting.
  bool feed(uint8_t byte);
  // Answer the waiting frame. clock_us returns a free-running microsecond counter.
  // Returns the number of response bytes available in tx(), or 0 when nothing is to be sent.
  size_t respond(uint32_t (*clock_us)());
  const uint8_t* tx() const { return tx_; }
  uint32_t crc_errors() const { return crc_errors_; }
  uint32_t length_errors() const { return length_errors_; }
  uint32_t ignored_frames() const { return ignored_; }

 private:
  size_t build(uint8_t type, uint8_t seq, const uint8_t* payload, size_t length);

  Engine* engine_;
  uint8_t state_;
  uint8_t header_[4];
  uint8_t header_len_;
  uint8_t payload_[kMaxPayload + 2];
  size_t need_;
  size_t have_;
  bool ready_;
  uint8_t type_;
  uint8_t seq_;
  uint16_t length_;
  uint8_t tx_[kMaxFrame];
  float in_[kHop];
  float out_[kHop];
  uint32_t crc_errors_;
  uint32_t length_errors_;
  uint32_t ignored_;
};

}  // namespace hushwave
