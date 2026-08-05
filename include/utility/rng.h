#pragma once
// Deterministic, seedable RNG shared by the simulation (maze generation,
// cheese placement) and the agent (exploration). A fixed seed reproduces the
// full trajectory; the generator state is persisted in checkpoints.

#include <cstdint>
#include <random>
#include <string>

namespace sir {

class Rng {
 public:
  explicit Rng(uint32_t seed);

  uint32_t nextU32();
  double uniform01();              // [0, 1)
  int uniformInt(int lo, int hi);  // inclusive both ends
  double uniform(double lo, double hi);

  // Portable full-state persistence via the standard stream operators.
  std::string saveState() const;
  bool restoreState(const std::string& state);

  uint32_t seed() const { return seed_; }

 private:
  std::mt19937 gen_;
  uint32_t seed_;
};

// Non-deterministic seed with fallback if random_device is not available.
uint32_t randomSeed();

}  // namespace sir
