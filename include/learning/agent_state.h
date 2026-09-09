#pragma once
// Flat, self-describing snapshot of the whole organism's learned and
// developmental state. Produced by Agent::exportState, consumed by
// CheckpointStore (save) and Agent::importState (load). Persisted in the
// checkpoint file; every vector is length-checked and finite-checked on load.

#include "organism/homeostasis.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sir {

struct AgentState {
  // --- topology ---
  int input = 0;
  int rnn = 0;
  int policy_out = 4;
  int pred_out = 16;

  // --- hyperparameters (informational; config governs at load) ---
  float lr = 0.0f;
  float gamma = 0.0f;
  float tau = 0.0f;

  // --- lifetime counters ---
  uint64_t training_steps = 0;
  uint64_t lifetime_steps = 0;
  uint64_t cheese_total = 0;
  uint32_t maze_generations = 0;
  uint32_t checkpoints_saved = 0;
  uint32_t invalid_updates = 0;
  uint64_t explored_count = 0;

  // --- exploration / randomness ---
  float epsilon = 0.0f;
  uint32_t seed = 0;
  int64_t timestamp_utc = 0;
  std::string rng_state;  // portable mt19937 state

  // --- organism internal state ---
  Homeostasis::Snapshot homeo;
  float novelty_ema = 0.0f;
  float pred_loss_ema = 0.0f;
  float uncertainty = 0.5f;

  // --- lifetime age / runtime ---
  uint64_t active_runtime_ms = 0;

  // --- consolidation history ---
  uint32_t consolidation_cycles = 0;
  uint64_t last_consolidation_step = 0;
  uint64_t consolidation_train_ops_total = 0;

  // --- structural plasticity history ---
  uint32_t structural_accepted = 0;
  uint32_t structural_rejected = 0;
  uint64_t pruned_total = 0;
  uint64_t rewired_total = 0;

  // --- neural parameters ---
  std::vector<float> online_params;
  std::vector<float> target_params;
  std::vector<float> adam_m;
  std::vector<float> adam_v;
  std::vector<uint8_t> masks;
  std::vector<float> utility;

  // --- bounded memories ---
  size_t episodic_capacity = 0;
  std::vector<float> episodic_floats;
  std::vector<uint32_t> episodic_meta;
  std::vector<uint32_t> novelty_table;
  std::vector<float> replay_floats;
  std::vector<uint32_t> replay_meta;

  bool allFinite() const;
};

inline bool AgentState::allFinite() const {
  auto finite_vec = [](const std::vector<float>& v) {
    for (float x : v)
      if (!std::isfinite(x)) return false;
    return true;
  };
  return finite_vec(online_params) && finite_vec(target_params) &&
         finite_vec(adam_m) && finite_vec(adam_v) && finite_vec(utility) &&
         finite_vec(episodic_floats) && finite_vec(replay_floats) &&
         std::isfinite(homeo.energy) && std::isfinite(homeo.hunger) &&
         std::isfinite(homeo.fatigue) && std::isfinite(homeo.stress) &&
         std::isfinite(homeo.curiosity_need) &&
         std::isfinite(homeo.satisfaction) && std::isfinite(homeo.uncertainty) &&
         std::isfinite(novelty_ema) && std::isfinite(pred_loss_ema) &&
         std::isfinite(uncertainty) && std::isfinite(epsilon);
}

}  // namespace sir
