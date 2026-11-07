#include "hushwave_protocol.h"

#include <string.h>

namespace hushwave {

namespace {
constexpr uint8_t kStateSync0 = 0;
constexpr uint8_t kStateSync1 = 1;
constexpr uint8_t kStateHeader = 2;
constexpr uint8_t kStateBody = 3;
constexpr char kName[] = "HushWave B5 float32";
}  // namespace

uint16_t crc16(const uint8_t* data, size_t length) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < length; ++i) {
    crc ^= static_cast<uint16_t>(data[i]) << 8;
    for (int b = 0; b < 8; ++b) {
      crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021) : static_cast<uint16_t>(crc << 1);
    }
  }
  return crc;
}

Link::Link(Engine* engine)
    : engine_(engine),
      state_(kStateSync0),
      header_len_(0),
      need_(0),
      have_(0),
      ready_(false),
      type_(0),
      seq_(0),
      length_(0),
      crc_errors_(0),
      length_errors_(0),
      ignored_(0) {}

bool Link::feed(uint8_t byte) {
  if (ready_) return true;  // the caller must respond() before feeding more
  switch (state_) {
    case kStateSync0:
      if (byte == kSync0) state_ = kStateSync1;
      break;
    case kStateSync1:
      if (byte == kSync1) {
        state_ = kStateHeader;
        header_len_ = 0;
      } else if (byte != kSync0) {
        state_ = kStateSync0;
      }
      break;
    case kStateHeader:
      header_[header_len_++] = byte;
      if (header_len_ == 4) {
        type_ = header_[0];
        seq_ = header_[1];
        length_ = static_cast<uint16_t>(header_[2] | (header_[3] << 8));
        if (length_ > kMaxPayload) {
          ++length_errors_;
          state_ = kStateSync0;
        } else {
          need_ = static_cast<size_t>(length_) + 2;
          have_ = 0;
          state_ = kStateBody;
        }
      }
      break;
    default:
      payload_[have_++] = byte;
      if (have_ == need_) {
        uint8_t body[4 + kMaxPayload];
        memcpy(body, header_, 4);
        memcpy(body + 4, payload_, length_);
        const uint16_t expect = crc16(body, 4 + static_cast<size_t>(length_));
        const uint16_t got = static_cast<uint16_t>(payload_[length_] | (payload_[length_ + 1] << 8));
        state_ = kStateSync0;
        if (expect == got) {
          ready_ = true;
        } else {
          ++crc_errors_;
        }
      }
      break;
  }
  return ready_;
}

size_t Link::build(uint8_t type, uint8_t seq, const uint8_t* payload, size_t length) {
  tx_[0] = kSync0;
  tx_[1] = kSync1;
  tx_[2] = type;
  tx_[3] = seq;
  tx_[4] = static_cast<uint8_t>(length & 0xFF);
  tx_[5] = static_cast<uint8_t>(length >> 8);
  if (length) memcpy(tx_ + 6, payload, length);
  const uint16_t crc = crc16(tx_ + 2, 4 + length);
  tx_[6 + length] = static_cast<uint8_t>(crc & 0xFF);
  tx_[7 + length] = static_cast<uint8_t>(crc >> 8);
  return 8 + length;
}

size_t Link::respond(uint32_t (*clock_us)()) {
  if (!ready_) return 0;
  ready_ = false;
  if (type_ == kTypeAudioIn && length_ == kAudioBytes) {
    for (int n = 0; n < kHop; ++n) {
      const int16_t v = static_cast<int16_t>(payload_[2 * n] | (payload_[2 * n + 1] << 8));
      in_[n] = static_cast<float>(v) / 32768.0f;
    }
    const uint32_t t0 = clock_us();
    engine_->process(in_, out_);
    uint32_t elapsed = clock_us() - t0;
    if (elapsed > 0xFFFF) elapsed = 0xFFFF;
    uint8_t reply[4 + kAudioBytes];
    reply[0] = static_cast<uint8_t>(elapsed & 0xFF);
    reply[1] = static_cast<uint8_t>(elapsed >> 8);
    reply[2] = 0;
    reply[3] = 0;
    for (int n = 0; n < kHop; ++n) {
      float v = out_[n] * 32768.0f;
      if (v > 32767.0f) v = 32767.0f;
      if (v < -32768.0f) v = -32768.0f;
      const int16_t q = static_cast<int16_t>(v >= 0.0f ? v + 0.5f : v - 0.5f);
      reply[4 + 2 * n] = static_cast<uint8_t>(static_cast<uint16_t>(q) & 0xFF);
      reply[5 + 2 * n] = static_cast<uint8_t>(static_cast<uint16_t>(q) >> 8);
    }
    return build(kTypeAudioOut, seq_, reply, sizeof(reply));
  }
  if (type_ == kTypeHello) {
    uint8_t reply[7 + sizeof(kName) - 1];
    reply[0] = kProtocolVersion;
    reply[1] = static_cast<uint8_t>(kSampleRate & 0xFF);
    reply[2] = static_cast<uint8_t>(kSampleRate >> 8);
    reply[3] = static_cast<uint8_t>(kHop & 0xFF);
    reply[4] = static_cast<uint8_t>(kHop >> 8);
    reply[5] = static_cast<uint8_t>(kDelay & 0xFF);
    reply[6] = static_cast<uint8_t>(kDelay >> 8);
    memcpy(reply + 7, kName, sizeof(kName) - 1);
    return build(kTypeHelloReply, seq_, reply, sizeof(reply));
  }
  if (type_ == kTypeReset) {
    engine_->reset();
    return build(kTypeResetAck, seq_, nullptr, 0);
  }
  ++ignored_;
  return 0;
}

}  // namespace hushwave
