#pragma once
// Lifelong development: continuous age-based schedules. All schedules are
// smooth functions of normalized age tau = min(1, steps / maturity_steps);
// there are no discrete life stages and learning never switches off.

#include <cstdint>

namespace sir {

struct DevParams {
  uint64_t maturity_steps = 200000;
  double lr_age_scale = 0.5;         // learning rate: 1 - scale*tau
  double epsilon_age_scale = 0.3;    // exploration: (1 - scale*tau) factor
  double plasticity_age_scale = 0.7; // structural change rate factor
};

class Development {
 public:
  explicit Development(const DevParams& p);

  // Normalized age in [0,1].
  double ageNorm(uint64_t lifetime_steps) const;

  // Continuous, monotone, bounded below (never zero) multipliers.
  double learningRateScale(uint64_t lifetime_steps) const;  // [0.1, 1]
  double epsilonScale(uint64_t lifetime_steps) const;       // [0.05, 1]
  double plasticityScale(uint64_t lifetime_steps) const;    // [0.1, 1]

 private:
  DevParams p_;
};

}  // namespace sir
