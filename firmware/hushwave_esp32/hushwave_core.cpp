#if defined(ARDUINO)
#pragma GCC optimize("O3")
#endif

#include "hushwave_core.h"

#include <math.h>
#include <string.h>

#include "hushwave_params.h"

namespace hushwave {

static_assert(kMinSubWindows == 6, "the sliding minimum ring holds five sub-windows");
static_assert(kRiseRatio > 1.0f, "the rise limit must be active");

namespace {
constexpr float kInf = INFINITY;
}

Engine::Engine() {
  const double pi = 3.14159265358979323846;
  for (int n = 0; n < kFrame; ++n) {
    window_[n] = static_cast<float>(sin(pi * n / kFrame));
  }
  for (int k = 0; k < kFrame / 2; ++k) {
    tw_re_[k] = static_cast<float>(cos(2.0 * pi * k / kFrame));
    tw_im_[k] = static_cast<float>(-sin(2.0 * pi * k / kFrame));
  }
  int bits = 0;
  while ((1 << bits) < kFrame) ++bits;
  for (int i = 0; i < kFrame; ++i) {
    int r = 0;
    for (int b = 0; b < bits; ++b) {
      if (i & (1 << b)) r |= 1 << (bits - 1 - b);
    }
    bitrev_[i] = static_cast<uint16_t>(r);
  }
  // Zero-phase second-order Butterworth high-pass magnitude at the bin centres.
  hp_mask_[0] = 0.0f;
  const double t_fc = tan(pi * kHpCutoffHz / kSampleRate);
  for (int k = 1; k < kBins - 1; ++k) {
    const double f = static_cast<double>(k) * kSampleRate / kFrame;
    const double ratio = t_fc / tan(pi * f / kSampleRate);
    hp_mask_[k] = static_cast<float>(1.0 / sqrt(1.0 + ratio * ratio * ratio * ratio));
  }
  hp_mask_[kBins - 1] = 1.0f;
  reset();
}

void Engine::reset() {
  memset(prev_, 0, sizeof(prev_));
  memset(overlap_, 0, sizeof(overlap_));
  memset(queue_, 0, sizeof(queue_));
  queue_len_ = kDelay;
  frames_ = 0;
  memset(ema_, 0, sizeof(ema_));
  memset(presence_, 0, sizeof(presence_));
  memset(num_, 0, sizeof(num_));
  memset(den_, 0, sizeof(den_));
  for (int u = 0; u < 5; ++u) {
    for (int k = 0; k < kBins; ++k) ring_[u][k] = kInf;
  }
  for (int k = 0; k < kBins; ++k) cur_min_[k] = kInf;
  ring_write_ = 0;
  sub_count_ = 0;
  alpha_pow_ = 1.0f;
  memset(rise_prev_, 0, sizeof(rise_prev_));
  have_prev_ = false;
  memset(prev_snr_, 0, sizeof(prev_snr_));
}

// In-place radix-2 decimation-in-time FFT of length kFrame (forward, e^{-j...}).
void Engine::fft(float* re, float* im) const {
  for (int i = 0; i < kFrame; ++i) {
    const int j = bitrev_[i];
    if (j > i) {
      const float tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const float ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (int len = 2; len <= kFrame; len <<= 1) {
    const int half = len >> 1;
    const int step = kFrame / len;
    for (int start = 0; start < kFrame; start += len) {
      for (int j = 0; j < half; ++j) {
        const float wr = tw_re_[j * step];
        const float wi = tw_im_[j * step];
        const int a = start + j;
        const int b = a + half;
        const float xr = re[b] * wr - im[b] * wi;
        const float xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
      }
    }
  }
}

// MCRA noise power tracker followed by the rise-rate limiter.
void Engine::update_noise(const float* power) {
  ++frames_;
  alpha_pow_ *= kMcraAlphaS;
  const float correction = 1.0f - alpha_pow_;
  for (int k = 0; k < kBins; ++k) {
    const float left = power[k == 0 ? 0 : k - 1];
    const float right = power[k == kBins - 1 ? kBins - 1 : k + 1];
    const float freq_smooth = 0.25f * left + 0.5f * power[k] + 0.25f * right;
    ema_[k] = kMcraAlphaS * ema_[k] + (1.0f - kMcraAlphaS) * freq_smooth;
    smooth_[k] = ema_[k] / correction;
  }
  const bool warm = frames_ > kMcraWarmupFrames;
  // Sliding minimum over kMinSubWindows sub-windows of kMinSubFrames frames each.
  if (warm) {
    for (int k = 0; k < kBins; ++k) {
      if (smooth_[k] < cur_min_[k]) cur_min_[k] = smooth_[k];
    }
    ++sub_count_;
  }
  for (int k = 0; k < kBins; ++k) {
    float indicator = 0.0f;
    if (warm) {
      float floor_of_smooth = cur_min_[k];
      if (sub_count_ == kMinSubFrames) {
        floor_of_smooth = kInf;
        for (int u = 0; u < kMinSubWindows - 1; ++u) {
          const float v = (u == ring_write_) ? cur_min_[k] : ring_[u][k];
          if (v < floor_of_smooth) floor_of_smooth = v;
        }
      } else {
        for (int u = 0; u < kMinSubWindows - 1; ++u) {
          if (ring_[u][k] < floor_of_smooth) floor_of_smooth = ring_[u][k];
        }
      }
      indicator = (smooth_[k] > kMcraDelta * floor_of_smooth) ? 1.0f : 0.0f;
    }
    presence_[k] = kMcraAlphaP * presence_[k] + (1.0f - kMcraAlphaP) * indicator;
    const float rate = kMcraAlphaD + (1.0f - kMcraAlphaD) * presence_[k];
    num_[k] = rate * num_[k] + (1.0f - rate) * power[k];
    den_[k] = rate * den_[k] + (1.0f - rate);
    float est = kMcraBias * num_[k] / den_[k];
    if (est < kNoiseFloor) est = kNoiseFloor;
    if (have_prev_) {
      const float limit = rise_prev_[k] * kRiseRatio;
      if (est > limit) est = limit;
    }
    rise_prev_[k] = est;
    noise_[k] = est;
  }
  if (warm && sub_count_ == kMinSubFrames) {
    for (int k = 0; k < kBins; ++k) {
      ring_[ring_write_][k] = cur_min_[k];
      cur_min_[k] = kInf;
    }
    ring_write_ = (ring_write_ + 1) % (kMinSubWindows - 1);
    sub_count_ = 0;
  }
  have_prev_ = true;
}

// Wiener gain with decision-directed a priori SNR, floor, and zero-phase high-pass.
void Engine::update_gain(const float* power) {
  for (int k = 0; k < kBins; ++k) {
    const float gamma = power[k] / noise_[k];
    const float ml = (gamma - 1.0f > 0.0f) ? gamma - 1.0f : 0.0f;
    const float xi = kDdAlpha * prev_snr_[k] + (1.0f - kDdAlpha) * ml;
    float g = xi / (1.0f + xi);
    if (g < kGainFloor) g = kGainFloor;
    if (g > 1.0f) g = 1.0f;
    prev_snr_[k] = g * g * gamma;
    gain_[k] = g * hp_mask_[k];
  }
}

void Engine::process(const float* in, float* out) {
  for (int n = 0; n < kHop; ++n) {
    re_[n] = prev_[n] * window_[n];
    re_[kHop + n] = in[n] * window_[kHop + n];
  }
  memset(im_, 0, sizeof(im_));
  memcpy(prev_, in, sizeof(prev_));
  fft(re_, im_);
  for (int k = 0; k < kBins; ++k) {
    power_[k] = re_[k] * re_[k] + im_[k] * im_[k];
  }
  update_noise(power_);
  update_gain(power_);
  // Inverse transform of the Hermitian spectrum G*Y through the conjugate trick:
  // x = conj(FFT(conj(Z))) / N, and only the real part is needed.
  for (int k = 0; k < kBins; ++k) {
    re_[k] *= gain_[k];
    im_[k] = -im_[k] * gain_[k];
  }
  for (int k = kBins; k < kFrame; ++k) {
    re_[k] = re_[kFrame - k];
    im_[k] = -im_[kFrame - k];
  }
  fft(re_, im_);
  const float scale = 1.0f / kFrame;
  float* seg = re_;
  for (int n = 0; n < kFrame; ++n) seg[n] = seg[n] * scale * window_[n];
  if (frames_ >= 2) {
    for (int n = 0; n < kHop; ++n) queue_[queue_len_ + n] = seg[n] + overlap_[n];
    queue_len_ += kHop;
  }
  memcpy(overlap_, seg + kHop, sizeof(overlap_));
  memcpy(out, queue_, kHop * sizeof(float));
  memmove(queue_, queue_ + kHop, (queue_len_ - kHop) * sizeof(float));
  queue_len_ -= kHop;
}

}  // namespace hushwave
