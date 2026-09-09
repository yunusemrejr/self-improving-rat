// Memory-layer tests: replay buffer (capacity, wraparound, prioritized
// sampling validity), episodic memory (capacity, significance replacement,
// serialization), novelty estimator (decay, bounds, capacity).

#include "test_framework.h"

#include "learning/replay_buffer.h"
#include "organism/episodic_memory.h"
#include "organism/novelty.h"
#include "utility/rng.h"

#include <vector>

using namespace sir;

TEST(replay_buffer_capacity_and_wraparound) {
  const int obs = 30, rnn = 16, cap = 64;
  ReplayBuffer rb(obs, rnn, cap);
  float s[30], s2[30], h[16], tgt[3];
  for (int i = 0; i < 500; ++i) {
    for (int k = 0; k < obs; ++k) { s[k] = static_cast<float>(i + k); s2[k] = static_cast<float>(i - k); }
    for (int k = 0; k < rnn; ++k) h[k] = static_cast<float>(k);
    tgt[0] = 0.1f; tgt[1] = 0.2f; tgt[2] = 0.3f;
    rb.push(s, s2, h, i % 4, static_cast<float>(i) * 0.01f, 0.5f, i % 7 == 0, tgt);
  }
  CHECK(rb.size() == static_cast<size_t>(cap));
  CHECK(rb.full());
  // Indices 0..size-1 are all valid and refer to the newest entries.
  for (int i = 0; i < cap; ++i) {
    const float* o = rb.obs(i);
    for (int k = 0; k < obs; ++k) CHECK(o[k] >= 0.0f);
    CHECK(rb.action(i) >= 0 && rb.action(i) < 4);
    CHECK(rb.reward(i) >= 0.0f);
    CHECK_NEAR(rb.extReward(i), 0.5f, 1e-6);
    const float* t = rb.targets(i);
    CHECK_NEAR(t[0], 0.1f, 1e-6);
    CHECK_NEAR(t[2], 0.3f, 1e-6);
  }
  rb.clear();
  CHECK(rb.empty());
}

TEST(replay_buffer_sampling_valid) {
  const int obs = 30, rnn = 16, cap = 100;
  ReplayBuffer rb(obs, rnn, cap);
  float s[30] = {0}, s2[30] = {0}, h[16] = {0}, tgt[3] = {0};
  for (int i = 0; i < cap; ++i) rb.push(s, s2, h, 0, 0.0f, 0.0f, false, tgt);
  Rng rng(1);
  std::vector<size_t> idx;
  rb.sampleIndices(rng, 1000, &idx);
  CHECK(idx.size() == 1000);
  for (size_t i : idx) CHECK(i < rb.size());
}

TEST(replay_buffer_is_weights_are_sampling_odds) {
  const int obs = 30, rnn = 16, cap = 100;
  ReplayBuffer rb(obs, rnn, cap);
  float s[30] = {0}, s2[30] = {0}, h[16] = {0}, tgt[3] = {0};
  // 99 low-priority entries + 1 cheese-like high-priority entry.
  for (int i = 0; i < cap - 1; ++i) rb.push(s, s2, h, 0, 0.0f, 0.0f, false, tgt);
  rb.push(s, s2, h, 1, 0.0f, 0.0f, false, tgt);
  rb.updatePriority(cap - 1, 100.0f);
  Rng rng(2);
  std::vector<size_t> idx;
  std::vector<float> w;
  double sum_high = 0.0, sum_low = 0.0;
  int n_high = 0, n_low = 0;
  for (int t = 0; t < 200; ++t) {
    rb.sampleIndices(rng, 100, &idx, &w);
    CHECK(idx.size() == w.size());
    for (size_t k = 0; k < idx.size(); ++k) {
      CHECK(w[k] > 0.0f);  // strictly positive odds
      if (idx[k] == static_cast<size_t>(cap - 1)) { sum_high += w[k]; ++n_high; }
      else { sum_low += w[k]; ++n_low; }
    }
  }
  // The high-priority entry is sampled far more often and with far larger
  // raw odds than the low-priority bulk.
  CHECK(n_high > 0 && n_low > 0);
  const double mean_high = sum_high / n_high;
  const double mean_low = sum_low / n_low;
  CHECK(mean_high > mean_low * 10.0);

  // With uniform priorities every raw odds is ~1 (w = N * (1/N) / 1).
  ReplayBuffer rb2(obs, rnn, 16);
  for (int i = 0; i < 16; ++i) rb2.push(s, s2, h, 0, 0.0f, 0.0f, false, tgt);
  Rng rng3(3);
  rb2.sampleIndices(rng3, 200, &idx, &w);
  for (float x : w) CHECK_NEAR(x, 1.0f, 1e-4);
}

TEST(sample_indices_optional_weights_does_not_change_draws) {
  const int obs = 30, rnn = 16, cap = 100;
  float s[30] = {0}, s2[30] = {0}, h[16] = {0}, tgt[3] = {0};
  // Mixed priorities so the distribution is not flat.
  ReplayBuffer rb_a(obs, rnn, cap);
  ReplayBuffer rb_b(obs, rnn, cap);
  for (int i = 0; i < cap; ++i) rb_a.push(s, s2, h, i % 4, 0.0f, 0.0f, false, tgt);
  for (int i = 0; i < cap; ++i) rb_b.push(s, s2, h, i % 4, 0.0f, 0.0f, false, tgt);
  rb_a.updatePriority(3, 50.0f);
  rb_a.updatePriority(cap - 1, 80.0f);
  rb_b.updatePriority(3, 50.0f);
  rb_b.updatePriority(cap - 1, 80.0f);
  std::vector<size_t> idx_a, idx_b;
  std::vector<float> w;
  Rng r1(42), r2(42);  // identical streams
  rb_a.sampleIndices(r1, 500, &idx_a);            // 3-arg form
  rb_b.sampleIndices(r2, 500, &idx_b, &w);        // 4-arg form
  CHECK(idx_a == idx_b);  // adding the optional output must not change draws
  CHECK(w.size() == idx_b.size());
  // Raw odds mean over draws equals N*sum(p^2)/(sum p)^2 >= 1 under the
  // sampling distribution (weights are odds, not a flat constant).
  double mean = 0.0;
  for (float x : w) mean += x;
  mean /= static_cast<double>(w.size());
  CHECK(mean >= 1.0 - 1e-3);
  for (float x : w) CHECK(x > 0.0f);
}

TEST(replay_buffer_prioritized_sampling_biases_toward_high_priority) {
  const int obs = 30, rnn = 16, cap = 100;
  ReplayBuffer rb(obs, rnn, cap);
  float s[30] = {0}, s2[30] = {0}, h[16] = {0}, tgt[3] = {0};
  // 10 low-priority + 1 high-priority entry.
  for (int i = 0; i < cap - 1; ++i) rb.push(s, s2, h, 0, 0.0f, 0.0f, false, tgt);
  rb.push(s, s2, h, 1, 0.0f, 0.0f, false, tgt);
  rb.updatePriority(cap - 1, 100.0f);  // the cheese-like entry
  Rng rng(2);
  std::vector<size_t> idx;
  int high_hits = 0;
  for (int t = 0; t < 200; ++t) {
    rb.sampleIndices(rng, 100, &idx);
    for (size_t i : idx)
      if (i == static_cast<size_t>(cap - 1)) ++high_hits;
  }
  // Priority 100 vs ~1e-3 floor: high entry must be sampled far more than
  // the uniform 1% rate.
  const double rate = static_cast<double>(high_hits) / (200 * 100);
  CHECK(rate > 0.1);
}

TEST(episodic_memory_capacity_and_significance_replacement) {
  EpisodicMemory mem(30, 16, 8);
  for (int i = 0; i < 8; ++i) {
    EpisodicEntry e;
    e.s.assign(30, static_cast<float>(i));
    e.s2.assign(30, static_cast<float>(i));
    e.h.assign(16, 0.0f);
    e.significance = 0.3f;
    e.insert_step = static_cast<uint64_t>(i);
    mem.add(std::move(e));
  }
  CHECK(mem.size() == 8);
  CHECK(mem.capacity() == 8);
  // Adding a high-significance entry replaces the lowest-significance one.
  EpisodicEntry high;
  high.s.assign(30, 99.0f);
  high.s2.assign(30, 99.0f);
  high.h.assign(16, 0.0f);
  high.significance = 3.0f;
  high.insert_step = 100;
  CHECK(mem.add(std::move(high)));  // replacement happened
  CHECK(mem.size() == 8);
  CHECK(mem.replacements() == 1);
  // All entries remain valid observations.
  for (size_t i = 0; i < mem.size(); ++i) {
    CHECK(mem[i].s.size() == 30);
    CHECK(mem[i].s2.size() == 30);
    CHECK(mem[i].h.size() == 16);
  }
}

TEST(episodic_memory_serialization_roundtrip) {
  EpisodicMemory mem(30, 16, 16);
  for (int i = 0; i < 5; ++i) {
    EpisodicEntry e;
    e.s.assign(30, static_cast<float>(i));
    e.s2.assign(30, static_cast<float>(i * 2));
    e.h.assign(16, static_cast<float>(i * 3));
    e.action = i % 4;
    e.reward = 1.0f + i;
    e.reward_ext = 0.5f + i * 0.1f;
    e.flags = i == 2 ? kFlagCheese : 0u;
    e.significance = 0.3f + i;
    e.insert_step = 1000 + i;
    mem.add(std::move(e));
  }
  std::vector<float> floats;
  std::vector<uint32_t> meta;
  mem.serializeTo(floats, meta);
  EpisodicMemory mem2(30, 16, 16);
  CHECK(mem2.restoreFrom(floats, meta));
  CHECK(mem2.size() == mem.size());
  for (size_t i = 0; i < mem.size(); ++i) {
    CHECK(mem2[i].s == mem[i].s);
    CHECK(mem2[i].s2 == mem[i].s2);
    CHECK(mem2[i].h == mem[i].h);
    CHECK(mem2[i].action == mem[i].action);
    CHECK_NEAR(mem2[i].reward, mem[i].reward, 1e-6);
    CHECK_NEAR(mem2[i].reward_ext, mem[i].reward_ext, 1e-6);
    CHECK(mem2[i].flags == mem[i].flags);
    CHECK(mem2[i].insert_step == mem[i].insert_step);
  }
  // Malformed restore is rejected.
  CHECK(!mem2.restoreFrom({1.0f, 2.0f}, {0, 0, 0}));
  CHECK(!mem2.restoreFrom(floats, {0}));
  // Restore beyond capacity is rejected.
  std::vector<uint32_t> big_meta(17 * 4, 0);
  std::vector<float> big_floats(17 * (30 * 2 + 16 + 3), 0.0f);
  CHECK(!mem2.restoreFrom(big_floats, big_meta));
}

TEST(episodic_memory_rejects_malformed_entries) {
  EpisodicMemory mem(30, 16, 8);
  EpisodicEntry bad;
  bad.s.assign(5, 0.0f);  // wrong dims
  bad.s2.assign(30, 0.0f);
  bad.h.assign(16, 0.0f);
  CHECK(!mem.add(std::move(bad)));
  CHECK(mem.empty());
}

TEST(novelty_decays_on_repeat_exposure) {
  NoveltyEstimator nv(64);
  float obs[30] = {0.25f, 0.5f, 0.75f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                   0.7f, 0.8f, 0.9f, 0.05f, 0.15f, 0.35f, 0.45f, 0.55f, 0.65f,
                   0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.1f, 0.2f,
                   0.3f, 0.4f};
  float first = nv.observe(obs, 30);
  CHECK_NEAR(first, 1.0f, 1e-6);  // first exposure is maximally novel
  float prev = first;
  for (int i = 0; i < 10; ++i) {
    const float n = nv.observe(obs, 30);
    CHECK(n >= 0.0f && n <= 1.0f);
    CHECK(n <= prev + 1e-6);  // novelty never increases on repeat exposure
    prev = n;
  }
  CHECK(prev < 0.5f);  // repeated exposure loses novelty
  CHECK(nv.used() >= 1);
}

TEST(novelty_table_bounded_capacity) {
  NoveltyEstimator nv(64);
  float obs[30] = {0};
  for (int i = 0; i < 10000; ++i) {
    // Many distinct observations -> table fills; capacity must stay bounded.
    obs[i % 30] = static_cast<float>((i % 97)) / 97.0f;
    nv.observe(obs, 30);
    CHECK(nv.used() <= nv.capacity());
  }
  // Serialization roundtrip preserves counts.
  std::vector<uint32_t> data;
  nv.serializeTo(data);
  NoveltyEstimator nv2(64);
  CHECK(nv2.restoreFrom(data));
  CHECK(nv2.used() == nv.used());
}
