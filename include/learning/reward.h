#pragma once

#include "simulation/observation.h"
#include "utility/config.h"

namespace sir {

// Potential-based reward from perception. Cheese collection terminates the
// current food-seeking episode, so the terminal potential is zero.
inline double scentPotentialReward(const Config& cfg, const float* s,
                                   const float* next, bool terminal) {
  const int frame = (cfg.observation_frames - 1) * kObservationBase;
  auto potential = [frame](const float* o) {
    return static_cast<double>(o[frame+8] + o[frame+9] + o[frame+10] + o[frame+11]);
  };
  return cfg.scent_proximity_reward_gain *
      ((terminal ? 0.0 : cfg.discount_factor * potential(next)) - potential(s));
}

}  // namespace sir
