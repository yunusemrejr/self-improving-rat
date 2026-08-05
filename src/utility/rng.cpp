#include "utility/rng.h"

#include <chrono>
#include <sstream>

namespace sir {

Rng::Rng(uint32_t seed) : gen_(seed), seed_(seed) {}

uint32_t Rng::nextU32() { return static_cast<uint32_t>(gen_()); }

double Rng::uniform01() {
  std::uniform_real_distribution<double> d(0.0, 1.0);
  return d(gen_);
}

int Rng::uniformInt(int lo, int hi) {
  std::uniform_int_distribution<int> d(lo, hi);
  return d(gen_);
}

double Rng::uniform(double lo, double hi) {
  std::uniform_real_distribution<double> d(lo, hi);
  return d(gen_);
}

std::string Rng::saveState() const {
  // The standard stream operators serialize the full mt19937 state in a
  // portable, round-trippable text form (624 state words + position).
  std::ostringstream os;
  os << gen_;
  return os.str();
}

bool Rng::restoreState(const std::string& state) {
  std::istringstream is(state);
  is >> gen_;
  return !is.fail() && !is.bad();
}

uint32_t randomSeed() {
  try {
    std::random_device rd;
    return rd();
  } catch (...) {
    // Fallback: time-based, non-deterministic enough for our purposes.
    auto t = std::chrono::high_resolution_clock::now().time_since_epoch();
    return static_cast<uint32_t>(t.count() & 0xFFFFFFFFu);
  }
}

}  // namespace sir
