#include "organism/episodic_memory.h"

#include <algorithm>
#include <cmath>

namespace sir {

EpisodicMemory::EpisodicMemory(int obs_dim, int rnn_dim, size_t capacity)
    : capacity_(capacity), obs_dim_(obs_dim), rnn_dim_(rnn_dim) {
  entries_.reserve(capacity_);
}

bool EpisodicMemory::add(EpisodicEntry e) {
  if (e.s.size() != static_cast<size_t>(obs_dim_) ||
      e.s2.size() != static_cast<size_t>(obs_dim_) ||
      e.h.size() != static_cast<size_t>(rnn_dim_)) {
    return false;  // defensive: reject malformed entries
  }
  if (entries_.size() < capacity_) {
    entries_.push_back(std::move(e));
    return false;
  }
  // Replace the least significant entry (ties: oldest).
  size_t worst = 0;
  float worst_score = entries_[0].significance;
  uint64_t worst_age = entries_[0].insert_step;
  for (size_t i = 1; i < entries_.size(); ++i) {
    const auto& en = entries_[i];
    if (en.significance < worst_score ||
        (en.significance == worst_score && en.insert_step < worst_age)) {
      worst = i;
      worst_score = en.significance;
      worst_age = en.insert_step;
    }
  }
  if (e.significance < worst_score) return false;
  entries_[worst] = std::move(e);
  ++replacements_;
  return true;
}

void EpisodicMemory::clear() { entries_.clear(); }

void EpisodicMemory::serializeTo(std::vector<float>& out_floats,
                                 std::vector<uint32_t>& out_meta) const {
  out_floats.clear();
  out_meta.clear();
  out_floats.reserve(entries_.size() * (obs_dim_ * 2 + rnn_dim_ + 3));
  out_meta.reserve(entries_.size() * 4);
  for (const auto& e : entries_) {
    out_floats.insert(out_floats.end(), e.s.begin(), e.s.end());
    out_floats.insert(out_floats.end(), e.s2.begin(), e.s2.end());
    out_floats.insert(out_floats.end(), e.h.begin(), e.h.end());
    out_floats.push_back(e.reward);
    out_floats.push_back(e.reward_ext);
    out_floats.push_back(e.significance);
    out_meta.push_back(static_cast<uint32_t>(e.action));
    out_meta.push_back(e.flags);
    out_meta.push_back(static_cast<uint32_t>(e.insert_step & 0xFFFFFFFFu));
    out_meta.push_back(static_cast<uint32_t>(e.insert_step >> 32));
  }
}

bool EpisodicMemory::restoreFrom(const std::vector<float>& floats,
                                 const std::vector<uint32_t>& meta) {
  const size_t per_entry = static_cast<size_t>(obs_dim_) * 2 + rnn_dim_ + 3;
  if (meta.size() % 4 != 0) return false;
  const size_t count = meta.size() / 4;
  if (count > capacity_) return false;
  if (floats.size() != count * per_entry) return false;
  for (float value : floats) if (!std::isfinite(value)) return false;
  for (size_t i = 0; i < count; ++i)
    if (meta[i * 4] > 3 || (meta[i * 4 + 1] & ~31u)) return false;
  entries_.clear();
  size_t f = 0;
  for (size_t i = 0; i < count; ++i) {
    EpisodicEntry e;
    e.s.assign(floats.begin() + static_cast<long>(f),
               floats.begin() + static_cast<long>(f + obs_dim_));
    f += obs_dim_;
    e.s2.assign(floats.begin() + static_cast<long>(f),
                floats.begin() + static_cast<long>(f + obs_dim_));
    f += obs_dim_;
    e.h.assign(floats.begin() + static_cast<long>(f),
               floats.begin() + static_cast<long>(f + rnn_dim_));
    f += rnn_dim_;
    e.reward = floats[f++];
    e.reward_ext = floats[f++];
    e.significance = floats[f++];
    e.action = static_cast<int>(meta[i * 4]);
    e.flags = meta[i * 4 + 1];
    e.insert_step = static_cast<uint64_t>(meta[i * 4 + 2]) |
                    (static_cast<uint64_t>(meta[i * 4 + 3]) << 32);
    entries_.push_back(std::move(e));
  }
  return true;
}

}  // namespace sir
