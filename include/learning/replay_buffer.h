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
  ReplayBuffer(int obs_dim, int rnn_dim, size_t capacity, float alpha = 1.0f);

  void push(const float* s, const float* s2, const float* h_prev, int action,
            float reward, float reward_ext, bool done, const float* homeo_targets);

  size_t size() const { return count_; }
  size_t capacity() const { return cap_; }
  bool empty() const { return count_ == 0; }
  bool full() const { return count_ == cap_; }
  void clear();
  void markBoundary();
  bool boundary(size_t i) const;
  float priority(size_t i) const;
  void serializeTo(std::vector<float>& values, std::vector<uint32_t>& meta) const;
  // Validates before changing anything; smaller capacities retain newest entries.
  bool restoreFrom(const std::vector<float>& values, const std::vector<uint32_t>& meta);


  // Prioritized sampling with replacement. New transitions start with the
  // current maximum priority; sampled transitions have their priority
  // updated by the learner via updatePriority(|TD error|). This makes rare
  // high-error experiences (cheese) train the network instead of being
  // diluted by the abundant low-error ones. Weights are proportional to
  // priority^alpha. When is_weights is non-null it receives, per drawn
  // index (same order as *out), the raw sampling odds
  //     w_raw_i = N * priority_i / sum(priority)        (N = count_)
  // The buffer applies no correction itself: the learner raises these raw
  // odds to the power (-beta) and max-normalizes them (config per_is_beta;
  // 0 disables the correction).
  void sampleIndices(Rng& rng, size_t batch, std::vector<size_t>* out,
                     std::vector<float>* is_weights = nullptr);
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

  void setPriority(size_t physical, float value);

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
  float alpha_;
  size_t leaves_ = 1;
  std::vector<double> tree_;
  float max_prio_ = 1.0f;
  static constexpr float kPrioMin = 1e-3f;
  static constexpr float kPrioMax = 1000.0f;
};

}  // namespace sir
