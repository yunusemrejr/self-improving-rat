#include "learning/replay_buffer.h"

#include <algorithm>
#include <cmath>

namespace sir {

ReplayBuffer::ReplayBuffer(int obs_dim, int rnn_dim, size_t capacity, float alpha)
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
      prio_(capacity, 1.0f), alpha_(std::clamp(alpha, 0.0f, 1.0f)) {
  while (leaves_ < cap_) leaves_ *= 2;
  tree_.assign(leaves_ * 2, 0.0);
}

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
  d_[idx(count_)] = done ? 3 : 0;
  r_[idx(count_)] = reward;
  setPriority(idx(count_), max_prio_);  // new transitions start with max priority
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
  std::fill(tree_.begin(), tree_.end(), 0.0);
}

void ReplayBuffer::sampleIndices(Rng& rng, size_t batch,
                                 std::vector<size_t>* out,
                                 std::vector<float>* is_weights) {
  out->clear();
  out->reserve(batch);
  if (is_weights) {
    is_weights->clear();
    is_weights->reserve(batch);
  }
  const size_t n = count_;
  if (n == 0) return;
  const double total = tree_[1];
  for (size_t b = 0; b < batch; ++b) {
    // Stratification reduces sampling variance; each draw descends O(log N).
    double u = (static_cast<double>(b) + rng.uniform01()) * total / batch;
    size_t leaf = 1;
    while (leaf < leaves_) {
      const double left = tree_[leaf * 2];
      if (u < left) leaf *= 2;
      else { u -= left; leaf = leaf * 2 + 1; }
    }
    const size_t physical = leaf - leaves_;
    out->push_back((physical + cap_ - head_) % cap_);
    if (is_weights)
      is_weights->push_back(static_cast<float>(n * tree_[leaf] / total));
  }
}

void ReplayBuffer::setPriority(size_t physical, float value) {
  prio_[physical] = value;
  size_t node = leaves_ + physical;
  tree_[node] = std::pow(static_cast<double>(value), alpha_);
  while (node > 1) {
    node /= 2;
    tree_[node] = tree_[node * 2] + tree_[node * 2 + 1];
  }
}

void ReplayBuffer::updatePriority(size_t index, float td_error) {
  if (index >= count_ || !std::isfinite(td_error)) return;
  float p = std::fabs(td_error);
  if (p < kPrioMin) p = kPrioMin;
  if (p > kPrioMax) p = kPrioMax;
  setPriority(idx(index), p);
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
bool ReplayBuffer::done(size_t i) const { return (d_[idx(i)] & 1) != 0; }

void ReplayBuffer::markBoundary() {
  if (count_) d_[idx(count_ - 1)] |= 2;
}
bool ReplayBuffer::boundary(size_t i) const { return (d_[idx(i)] & 2) != 0; }
float ReplayBuffer::priority(size_t i) const { return prio_[idx(i)]; }

void ReplayBuffer::serializeTo(std::vector<float>& values, std::vector<uint32_t>& meta) const {
  values.clear(); meta.clear();
  // Bound disk snapshots to 8 MiB even when the runtime replay is much larger.
  const size_t stride = 2 * obs_dim_ + rnn_dim_ + 6;
  const size_t saved = std::min({count_, size_t(65536), size_t(8 * 1024 * 1024) / (stride * 4 + 8)});
  values.reserve(saved * stride);
  meta.reserve(saved * 2);
  for (size_t i = count_ - saved; i < count_; ++i) {
    values.insert(values.end(), obs(i), obs(i) + obs_dim_);
    values.insert(values.end(), next_obs(i), next_obs(i) + obs_dim_);
    values.insert(values.end(), h(i), h(i) + rnn_dim_);
    values.insert(values.end(), targets(i), targets(i) + 3);
    values.push_back(extReward(i)); values.push_back(reward(i)); values.push_back(priority(i));
    meta.push_back(static_cast<uint32_t>(action(i))); meta.push_back(d_[idx(i)]);
  }
}

bool ReplayBuffer::restoreFrom(const std::vector<float>& values, const std::vector<uint32_t>& meta) {
  const size_t stride = 2 * obs_dim_ + rnn_dim_ + 6;
  if (meta.size() % 2 || meta.size() / 2 > 65536 || values.size() != meta.size() / 2 * stride)
    return false;
  for (float v : values) if (!std::isfinite(v)) return false;
  const size_t count = meta.size() / 2;
  for (size_t i = 0; i < count; ++i) {
    const float pr = values[(i + 1) * stride - 1];
    if (meta[i * 2] > 3 || meta[i * 2 + 1] > 3 || pr < kPrioMin || pr > kPrioMax)
      return false;
  }
  clear();
  const size_t first = count > cap_ ? count - cap_ : 0;
  for (size_t i = first; i < count; ++i) {
    const float* v = values.data() + i * stride;
    const float* t = v + 2 * obs_dim_ + rnn_dim_;
    push(v, v + obs_dim_, v + 2 * obs_dim_, meta[i * 2], t[4], t[3],
         (meta[i * 2 + 1] & 1) != 0, t);
    d_[idx(count_ - 1)] = static_cast<uint8_t>(meta[i * 2 + 1]);
    updatePriority(count_ - 1, t[5]);
  }
  return true;
}

}  // namespace sir
