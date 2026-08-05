#pragma once
// Fixed-capacity ring buffer of transitions
//   (s, s2, h_prev, action, reward, done, homeostatic targets)
// for experience replay. Preallocated; never grows.

#include "utility/rng.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sir {

class ReplayBuffer {
 public:
  ReplayBuffer(int obs_dim, int rnn_dim, size_t capacity);

  void push(const float* s, const float* s2, const float* h_prev, int action,
            float reward, float reward_ext, bool done, const float* homeo_targets);

  size_t size() const { return count_; }
  size_t capacity() const { return cap_; }
  bool empty() const { return count_ == 0; }
  bool full() const { return count_ == cap_; }
  void clear();

  // Prioritized sampling with replacement. New transitions start with the
  // current maximum priority; sampled transitions have their priority
  // updated by the learner via updatePriority(|TD error|). This makes rare
  // high-error experiences (cheese) train the network instead of being
  // diluted by the abundant low-error ones. Weights are proportional to
  // priority (power 1.0); no importance-sampling correction is applied
  // (documented approximation; the policy is near-stationary).
  void sampleIndices(Rng& rng, size_t batch, std::vector<size_t>* out);
  void updatePriority(size_t index, float td_error);

  // Random access (index in [0, size())).
  const float* obs(size_t i) const;
  const float* next_obs(size_t i) const;
  const float* h(size_t i) const;
  const float* targets(size_t i) const;   // 3 homeostatic target deltas
  float extReward(size_t i) const;        // external reward (prediction target)
  int action(size_t i) const;
  float reward(size_t i) const;           // composite reward (RL target)
  bool done(size_t i) const;

  int obsDim() const { return obs_dim_; }
  int rnnDim() const { return rnn_dim_; }

 private:
  size_t idx(size_t i) const { return (head_ + i) % cap_; }

  int obs_dim_;
  int rnn_dim_;
  size_t cap_;
  size_t head_ = 0;
  size_t count_ = 0;
  // tgt_: 3 homeostatic targets + 1 external reward per entry (used as the
  // world-model training target).
  std::vector<float> s_, s2_, h_, tgt_;
  std::vector<uint8_t> a_, d_;
  std::vector<float> r_;
  std::vector<float> prio_;  // per-entry sampling priority (>= kPrioMin)
  float max_prio_ = 1.0f;
  static constexpr float kPrioMin = 1e-3f;
  static constexpr float kPrioMax = 1000.0f;
};

}  // namespace sir
