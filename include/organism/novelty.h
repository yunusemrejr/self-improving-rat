#pragma once
// Bounded novelty estimation. Observations are quantized into coarse codes;
// a fixed-capacity hash table counts visits per code. Novelty = 1/sqrt(count)
// so repeated exposure to the same state reduces its novelty value. The table
// is persisted in checkpoints ("novelty-estimation state").

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sir {

class NoveltyEstimator {
 public:
  explicit NoveltyEstimator(size_t capacity = 1024);

  // Quantizes an observation into a 32-bit code (see novelty.cpp).
  static uint32_t code(const float* obs, int dim);

  // Returns novelty in [0,1] for the quantized state and increments its count.
  float observe(const float* obs, int dim);

  // Persistence: out receives pairs (code, count) for used slots.
  void serializeTo(std::vector<uint32_t>& out) const;
  bool restoreFrom(const std::vector<uint32_t>& data);

  size_t capacity() const { return slots_.size(); }
  size_t used() const { return used_; }

 private:
  struct Slot {
    uint32_t code = 0;
    uint32_t count = 0;  // bounded to kMaxCount
    bool used = false;
  };
  static constexpr uint32_t kMaxCount = 1000000;

  size_t findSlot(uint32_t code) const;  // index or capacity() if full
  void compact();  // halve all counts, drop zeros (bounded memory)

  std::vector<Slot> slots_;
  size_t used_ = 0;
};

}  // namespace sir
