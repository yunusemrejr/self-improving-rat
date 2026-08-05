#include "learning/replay_buffer.h"

#include <algorithm>

namespace sir {

ReplayBuffer::ReplayBuffer(int obs_dim, int rnn_dim, size_t capacity)
    : obs_dim_(obs_dim),
      rnn_dim_(rnn_dim),
      cap_(capacity),
      s_(capacity * obs_dim, 0.0f),
      s2_(capacity * obs_dim, 0.0f),
      h_(capacity * rnn_dim, 0.0f),
      tgt_(capacity * 4, 0.0f),
      a_(capacity, 0),
      d_(capacity, 0),
      r_(capacity, 0.0f),
      prio_(capacity, 1.0f) {}

void ReplayBuffer::push(const float* s, const float* s2, const float* h_prev,
                        int action, float reward, float reward_ext, bool done,
                        const float* homeo_targets) {
  std::copy(s, s + obs_dim_, s_.begin() + static_cast<long>(idx(count_)) * obs_dim_);
  std::copy(s2, s2 + obs_dim_, s2_.begin() + static_cast<long>(idx(count_)) * obs_dim_);
  std::copy(h_prev, h_prev + rnn_dim_, h_.begin() + static_cast<long>(idx(count_)) * rnn_dim_);
  std::copy(homeo_targets, homeo_targets + 3,
            tgt_.begin() + static_cast<long>(idx(count_)) * 4);
  tgt_[static_cast<long>(idx(count_)) * 4 + 3] = reward_ext;
  a_[idx(count_)] = static_cast<uint8_t>(action);
  d_[idx(count_)] = done ? 1 : 0;
  r_[idx(count_)] = reward;
  prio_[idx(count_)] = max_prio_;  // new transitions start with max priority
  if (count_ < cap_) {
    ++count_;
  } else {
    head_ = (head_ + 1) % cap_;
  }
}

void ReplayBuffer::clear() {
  head_ = 0;
  count_ = 0;
  max_prio_ = 1.0f;
}

void ReplayBuffer::sampleIndices(Rng& rng, size_t batch,
                                 std::vector<size_t>* out) {
  out->clear();
  out->reserve(batch);
  const size_t n = count_;
  if (n == 0) return;
  // Build the cumulative priority distribution once per call.
  std::vector<double> cum(n);
  double total = 0.0;
  for (size_t i = 0; i < n; ++i) {
    total += static_cast<double>(prio_[idx(i)]);
    cum[i] = total;
  }
  for (size_t b = 0; b < batch; ++b) {
    const double u = rng.uniform01() * total;
    // Binary search over the cumulative distribution.
    size_t lo = 0, hi = n - 1;
    while (lo < hi) {
      const size_t mid = (lo + hi) / 2;
      if (cum[mid] < u) lo = mid + 1;
      else hi = mid;
    }
    out->push_back(lo);
  }
}

void ReplayBuffer::updatePriority(size_t index, float td_error) {
  if (index >= count_) return;
  float p = std::fabs(td_error);
  if (p < kPrioMin) p = kPrioMin;
  if (p > kPrioMax) p = kPrioMax;
  prio_[idx(index)] = p;
  if (p > max_prio_) max_prio_ = p;
}

const float* ReplayBuffer::obs(size_t i) const {
  return s_.data() + idx(i) * obs_dim_;
}
const float* ReplayBuffer::next_obs(size_t i) const {
  return s2_.data() + idx(i) * obs_dim_;
}
const float* ReplayBuffer::h(size_t i) const {
  return h_.data() + idx(i) * rnn_dim_;
}
const float* ReplayBuffer::targets(size_t i) const {
  return tgt_.data() + idx(i) * 4;
}
float ReplayBuffer::extReward(size_t i) const { return tgt_[idx(i) * 4 + 3]; }
int ReplayBuffer::action(size_t i) const { return a_[idx(i)]; }
float ReplayBuffer::reward(size_t i) const { return r_[idx(i)]; }
bool ReplayBuffer::done(size_t i) const { return d_[idx(i)] != 0; }

}  // namespace sir
