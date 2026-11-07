// HushWave DSP core: the frozen B5 enhancer in single precision.
// Wiener gain with decision-directed SNR, MCRA noise tracking with a rise-rate limit,
// zero-phase spectral high-pass, square-root Hann windows, 50 percent overlap-add.
// Plain C++ with no Arduino dependency, so the same code runs on the ESP32 and on a PC.
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace hushwave {

constexpr int kSampleRate = 16000;
constexpr int kFrame = 512;
constexpr int kHop = 256;
constexpr int kBins = kFrame / 2 + 1;
constexpr int kDelay = kFrame - 1;  // fixed algorithmic delay in samples

class Engine {
 public:
  Engine();
  // Forget the noise estimate and all buffers.
  void reset();
  // Process exactly kHop samples in [-1, 1]; writes kHop samples (delayed by kDelay).
  void process(const float* in, float* out);

 private:
  void fft(float* re, float* im) const;
  void update_noise(const float* power);
  void update_gain(const float* power);

  // tables
  float window_[kFrame];
  float tw_re_[kFrame / 2];
  float tw_im_[kFrame / 2];
  uint16_t bitrev_[kFrame];
  float hp_mask_[kBins];

  // stream state
  float prev_[kHop];
  float overlap_[kHop];
  float queue_[kDelay + kHop];
  int queue_len_;
  int frames_;

  // MCRA state
  float ema_[kBins];
  float presence_[kBins];
  float num_[kBins];
  float den_[kBins];
  float ring_[5][kBins];
  float cur_min_[kBins];
  int ring_write_;
  int sub_count_;
  float alpha_pow_;
  float rise_prev_[kBins];
  bool have_prev_;

  // gain state
  float prev_snr_[kBins];

  // work buffers
  float re_[kFrame];
  float im_[kFrame];
  float power_[kBins];
  float noise_[kBins];
  float gain_[kBins];
  float smooth_[kBins];
};

}  // namespace hushwave
