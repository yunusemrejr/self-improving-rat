#pragma once
// Strictly bounded episodic memory. Stores a fixed capacity of compact,
// significant transitions (observation, next observation, recurrent state,
// action, reward, event flags, significance). When full, the least
// significant entry is replaced (ties: oldest first). Never grows unbounded.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sir {

// Event flags (bitfield).
enum EpisodicFlag : uint32_t {
  kFlagCheese = 1u << 0,
  kFlagWall = 1u << 1,
  kFlagHighNovelty = 1u << 2,
  kFlagHighStress = 1u << 3,
  kFlagHighPredError = 1u << 4,
};

struct EpisodicEntry {
  std::vector<float> s;    // observation before action (obs_dim)
  std::vector<float> s2;   // observation after action (obs_dim)
  std::vector<float> h;    // recurrent state before action (rnn_dim)
  int action = 0;
  float reward = 0.0f;     // composite reward
  float reward_ext = 0.0f; // external reward (world-model target)
  uint32_t flags = 0;
  float significance = 0.0f;
  uint64_t insert_step = 0;
};

class EpisodicMemory {
 public:
  EpisodicMemory(int obs_dim, int rnn_dim, size_t capacity);

  // Adds a significant transition; replaces the least significant entry when
  // full. Returns true if an existing entry was replaced.
  bool add(EpisodicEntry e);

  size_t size() const { return entries_.size(); }
  size_t capacity() const { return capacity_; }
  size_t replacements() const { return replacements_; }
  bool empty() const { return entries_.empty(); }
  const EpisodicEntry& operator[](size_t i) const { return entries_[i]; }
  void clear();

  // Flat serialization for checkpoints: floats = [s | s2 | h | reward |
  // reward_ext | significance] per entry; meta = [action | flags |
  // insert_step_lo | insert_step_hi] per entry.
  void serializeTo(std::vector<float>& out_floats,
                   std::vector<uint32_t>& out_meta) const;
  bool restoreFrom(const std::vector<float>& floats,
                   const std::vector<uint32_t>& meta);

 private:
  size_t capacity_;
  std::vector<EpisodicEntry> entries_;
  size_t replacements_ = 0;
  int obs_dim_;
  int rnn_dim_;
};

}  // namespace sir
