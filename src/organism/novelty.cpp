#include "organism/novelty.h"

#include <cmath>

namespace sir {

namespace {
// splitmix-style mixing for the code hash.
uint32_t mix(uint32_t x) {
  x += 0x9e3779b9u;
  x = (x ^ (x >> 16)) * 0x85ebca6bu;
  x = (x ^ (x >> 13)) * 0xc2b2ae35u;
  x ^= x >> 16;
  return x;
}
}  // namespace

NoveltyEstimator::NoveltyEstimator(size_t capacity) : slots_(capacity) {}

uint32_t NoveltyEstimator::code(const float* obs, int dim) {
  // Quantize the first min(dim, 30) channels: each channel contributes bits:
  //   wall bits (8)          -> 1 bit each
  //   scent channels (4)     -> 1 bit (>= 0.25)
  //   last action (4)        -> 2 bits (action index)
  //   wall-hit flag          -> 1 bit
  //   revisit signal         -> 1 bit
  //   homeostasis (6 first)  -> 2 bits each (buckets)
  //   novelty channel        -> 1 bit
  //   resting flag           -> 1 bit
  uint32_t c = 0;
  int bit = 0;
  auto put = [&](uint32_t v, int nbits) {
    c |= (v & ((1u << nbits) - 1)) << bit;
    bit += nbits;
  };
  if (dim >= 8) {
    for (int i = 0; i < 8; ++i) put(obs[i] > 0.5f ? 1 : 0, 1);
  }
  if (dim >= 12) {
    for (int i = 8; i < 12; ++i) put(obs[i] > 0.25f ? 1 : 0, 1);
  }
  if (dim >= 16) {
    uint32_t a = 0;
    for (int i = 12; i < 16; ++i)
      if (obs[i] > 0.5f) a = static_cast<uint32_t>(i - 12);
    put(a, 2);
  }
  if (dim >= 17) put(obs[16] > 0.5f ? 1 : 0, 1);
  if (dim >= 18) put(obs[17] > 0.25f ? 1 : 0, 1);
  if (dim >= 25) {
    for (int i = 19; i < 25; ++i) {
      uint32_t b = static_cast<uint32_t>(obs[i] * 3.999f);
      put(b > 3 ? 3 : b, 2);
    }
  }
  if (dim >= 27) put(obs[26] > 0.25f ? 1 : 0, 1);
  if (dim >= 28) put(obs[27] > 0.5f ? 1 : 0, 1);
  return c;
}

size_t NoveltyEstimator::findSlot(uint32_t code) const {
  const size_t cap = slots_.size();
  if (cap == 0) return cap;
  size_t i = mix(code) % cap;
  for (size_t probe = 0; probe < cap; ++probe) {
    const Slot& s = slots_[i];
    if (!s.used || s.code == code) return i;
    i = (i + 1) % cap;
  }
  return cap;  // full
}

void NoveltyEstimator::compact() {
  for (auto& s : slots_) {
    if (!s.used) continue;
    s.count = (s.count + 1) / 2;
    if (s.count == 0) {
      s.used = false;
      --used_;
    }
  }
}

float NoveltyEstimator::observe(const float* obs, int dim) {
  const uint32_t c = code(obs, dim);
  size_t i = findSlot(c);
  if (i == slots_.size()) {
    compact();
    i = findSlot(c);
    if (i == slots_.size()) return 0.0f;  // pathological; no unbounded growth
  }
  Slot& s = slots_[i];
  if (!s.used) {
    s.used = true;
    s.code = c;
    s.count = 1;
    ++used_;
    return 1.0f;
  }
  if (s.count < kMaxCount) ++s.count;
  // novelty = 1/sqrt(count): first exposure 1.0, then decays.
  return static_cast<float>(1.0 / std::sqrt(static_cast<double>(s.count)));
}

void NoveltyEstimator::serializeTo(std::vector<uint32_t>& out) const {
  out.clear();
  out.reserve(used_ * 2);
  for (const auto& s : slots_) {
    if (s.used) {
      out.push_back(s.code);
      out.push_back(s.count);
    }
  }
}

bool NoveltyEstimator::restoreFrom(const std::vector<uint32_t>& data) {
  for (auto& s : slots_) s = Slot{};
  used_ = 0;
  if (data.size() % 2 != 0) return false;
  for (size_t i = 0; i < data.size(); i += 2) {
    size_t idx = findSlot(data[i]);
    if (idx == slots_.size()) {
      compact();
      idx = findSlot(data[i]);
      if (idx == slots_.size()) return false;
    }
    Slot& s = slots_[idx];
    s.used = true;
    s.code = data[i];
    s.count = data[i + 1];
    ++used_;
  }
  return true;
}

}  // namespace sir
