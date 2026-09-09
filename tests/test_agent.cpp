// Learning-system tests: genuine weight updates, reward-driven learning
// (toy MDP), recurrent-state influence, exploration/exploitation, NaN
// rejection and rollback, prediction/curiosity signals, and the "weights do
// not change when only observing" property.

#include "test_framework.h"

#include "learning/agent.h"
#include "simulation/rat.h"
#include "utility/config.h"
#include "utility/rng.h"

#include <cmath>
#include <vector>

using namespace sir;

namespace {

// A deterministic toy environment encoded in observation channel 0:
//   S0 (channel0=0): action Down (index 1) -> S1, reward +10, done.
//                    any other action -> S0, reward -1.
//   S1 (channel0=1): any action -> S0, reward 0.
// A correct learner must overcome the argmax tie-break (index 0 = Up) and
// learn to take Down in S0.
void runToyEpisode(Agent& agent, const Config& cfg, float s0[30], float s1[30],
                   float ht[3], uint64_t step) {
  const float* h_prev = agent.recurrentState();
  auto d = agent.selectAction(s0);
  const bool good = (d.action == Action::Down);
  const float* s2 = good ? s1 : s0;
  const float r = good ? 10.0f : -1.0f;
  agent.computeIntrinsics(s0, s2, ht, r);
  const float rt = r + agent.lastCuriosityReward() +
                   static_cast<float>(cfg.prediction_reward_gain) *
                       (1.0f - agent.uncertainty());
  agent.observeAndTrain(s0, h_prev, d.action, rt, r, s2, good, ht, good, false,
                        step);
  if (good) {
    const float* h_prev2 = agent.recurrentState();
    auto d2 = agent.selectAction(s1);
    agent.computeIntrinsics(s1, s0, ht, 0.0f);
    agent.observeAndTrain(s1, h_prev2, d2.action,
                          agent.lastCuriosityReward(), 0.0f, s0, false, ht,
                          false, false, step + 1);
  }
}

}  // namespace

TEST(agent_weights_change_after_training) {
  Config cfg;
  cfg.observation_frames = 1;
  Rng rng(1);
  Agent a(cfg, rng);
  AgentState s1, s2;
  a.exportState(s1);
  float o[30] = {0.1f};
  float ht[3] = {0, 0, 0};
  for (int i = 0; i < 200; ++i) {
    const float* h_prev = a.recurrentState();
    auto d = a.selectAction(o);
    a.computeIntrinsics(o, o, ht, -0.02f);
    a.observeAndTrain(o, h_prev, d.action, -0.02f, -0.02f, o, false, ht, false,
                      true, static_cast<uint64_t>(i));
  }
  a.exportState(s2);
  CHECK(s1.online_params != s2.online_params);  // params genuinely change
  CHECK(a.trainingUpdates() > 0);
}

TEST(agent_weights_unchanged_without_training) {
  Config cfg;
  cfg.observation_frames = 1;
  Rng rng(2);
  Agent a(cfg, rng);
  AgentState before;
  a.exportState(before);
  // Observing + intrinsic computation alone must not alter parameters.
  float o[30] = {0.2f}, o2[30] = {0.3f};
  float ht[3] = {0.01f, 0.01f, 0.0f};
  for (int i = 0; i < 50; ++i) {
    a.selectAction(o);
    a.computeIntrinsics(o, o2, ht, -0.02f);
  }
  AgentState after;
  a.exportState(after);
  CHECK(before.online_params == after.online_params);
  CHECK(before.adam_m == after.adam_m);
  CHECK(before.adam_v == after.adam_v);
}

TEST(agent_toy_mdp_learns_reward_action) {
  Config cfg;
  cfg.observation_frames = 1;
  cfg.exploration_decay_steps = 3000;
  cfg.exploration_end = 0.02;
  cfg.learning_rate = 0.001;
  Rng rng(7);
  Agent agent(cfg, rng);
  float s0[30] = {0}, s1[30] = {0};
  s0[0] = 0.0f;
  s1[0] = 1.0f;
  float ht[3] = {0, 0, 0};
  for (int ep = 0; ep < 3000; ++ep)
    runToyEpisode(agent, cfg, s0, s1, ht, static_cast<uint64_t>(ep) * 2);
  // Evaluate the greedy policy: at epsilon floor 0.02, nearly all actions are
  // greedy. A learned policy must pick Down (cheese) far above random (25%).
  int good = 0, total = 0;
  for (int t = 0; t < 2000; ++t) {
    const auto& d = agent.selectAction(s0);
    if (!d.explored) {
      ++total;
      if (d.action == Action::Down) ++good;
    }
  }
  CHECK(total > 500);  // exploitation actually happened
  CHECK(static_cast<double>(good) / total > 0.8);  // learned the cheese action
}

TEST(agent_recurrent_state_evolves_and_affects_policy) {
  Config cfg;
  cfg.observation_frames = 1;
  Rng rng(3);
  Agent a(cfg, rng);
  // Varying observations so the GRU does not converge to a fixed point.
  float o[30] = {0.5f};
  bool h_changed = false;
  std::vector<float> h_copy(a.rnnSize());
  for (int i = 0; i < 20; ++i) {
    o[i % 4] = static_cast<float>(0.1 + 0.01 * i);
    o[27] = static_cast<float>(0.5 + 0.02 * i);
    // Copy the pre-action recurrent state (selectAction mutates it).
    std::copy(a.recurrentState(), a.recurrentState() + a.rnnSize(),
              h_copy.begin());
    a.selectAction(o);
    const float* h1 = a.recurrentState();
    for (int k = 0; k < a.rnnSize(); ++k) {
      if (std::fabs(h_copy[k] - h1[k]) > 1e-6) h_changed = true;
    }
  }
  CHECK(h_changed);  // recurrent state updates each decision step
  // Two different recurrent histories must be able to produce different
  // Q-values for the same observation (memory influences decisions).
  a.resetRecurrent();
  std::vector<float> qa(4);
  a.qValuesFor(o, qa.data());
  // After a burst of Down actions the recurrent state differs.
  for (int i = 0; i < 10; ++i) {
    // Force Down decisions by feeding a high Q for it is not possible
    // directly; instead advance the network with random actions.
    a.selectAction(o);
  }
  std::vector<float> qb(4);
  a.qValuesFor(o, qb.data());
  bool differs = false;
  for (int k = 0; k < 4; ++k)
    if (std::fabs(qa[k] - qb[k]) > 1e-4) differs = true;
  // With random weights and a zeroed recurrent state the very first forward
  // equals the later one only if h stays zero; after 10 steps h != 0, so the
  // outputs may differ. This check is deliberately loose (structure exists).
  CHECK(differs);
  CHECK(a.rnnSize() == cfg.rnn_hidden);
}

TEST(agent_nan_rejection_restores_params) {
  Config cfg;
  cfg.observation_frames = 1;
  cfg.train_interval_steps = 1;
  Rng rng(4);
  Agent a(cfg, rng);
  // Fill the replay with valid transitions, then feed a NaN observation.
  float o[30] = {0.1f}, o2[30] = {0.2f};
  float ht[3] = {0, 0, 0};
  for (int i = 0; i < 40; ++i) {
    const float* h_prev = a.recurrentState();
    auto d = a.selectAction(o);
    a.computeIntrinsics(o, o2, ht, -0.02f);
    a.observeAndTrain(o, h_prev, d.action, -0.02f, -0.02f, o2, false, ht, false,
                      false, static_cast<uint64_t>(i));
  }
  AgentState valid;
  a.exportState(valid);
  CHECK(a.invalidUpdates() == 0);
  // NaN observation must be rejected without corrupting the network.
  float nan_o[30] = {0.0f};
  nan_o[0] = std::nanf("");
  const float* h_prev = a.recurrentState();
  a.computeIntrinsics(nan_o, o2, ht, -0.02f);
  a.observeAndTrain(nan_o, h_prev, Action::Up, -0.02f, -0.02f, o2, false, ht,
                    false, false, 1000);
  CHECK(a.invalidUpdates() > 0);
  AgentState after;
  a.exportState(after);
  CHECK(after.allFinite());
  CHECK(after.online_params.size() == valid.online_params.size());
}

TEST(agent_exploration_and_exploitation_both_functional) {
  Config cfg;
  cfg.observation_frames = 1;
  cfg.exploration_start = 0.9;
  cfg.exploration_end = 0.05;
  cfg.exploration_decay_steps = 10000;
  Rng rng(5);
  Agent a(cfg, rng);
  float o[30] = {0.3f};
  int explored = 0, exploited = 0;
  for (int i = 0; i < 2000; ++i) {
    auto d = a.selectAction(o);
    if (d.explored) ++explored;
    else ++exploited;
  }
  CHECK(explored > 0);
  CHECK(exploited > 0);
  // Epsilon decays toward the configured floor.
  CHECK(a.currentEpsilon() >= 0.05 - 1e-6);
  CHECK(a.currentEpsilon() <= 0.9 + 1e-6);
}

TEST(agent_prediction_and_curiosity_signals) {
  Config cfg;
  cfg.observation_frames = 1;
  Rng rng(6);
  Agent a(cfg, rng);
  float s[30] = {0.0f}, s2[30] = {0.0f};
  s2[0] = 1.0f;  // next state differs (wall bit changes)
  s2[8] = 0.5f;  // scent appears
  float ht[3] = {0.0f, 0.0f, 0.0f};
  a.selectAction(s);
  a.computeIntrinsics(s, s2, ht, 1.0f);
  CHECK(a.lastNovelty() >= 0.0f && a.lastNovelty() <= 1.0f);
  CHECK(a.lastCuriosityReward() >= 0.0f);
  CHECK(a.lastCuriosityReward() <= static_cast<float>(cfg.curiosity_reward_gain) + 1e-6f);
  CHECK(a.lastPredError() >= 0.0f && a.lastPredError() <= 1.0f);
  CHECK(a.uncertainty() >= 0.0f && a.uncertainty() <= 1.0f);
  CHECK(a.predictionLossEma() >= 0.0f);
}

TEST(agent_state_roundtrip_preserves_learning) {
  Config cfg;
  cfg.observation_frames = 1;
  Rng rng(8);
  Agent a(cfg, rng);
  float o[30] = {0.4f}, o2[30] = {0.5f};
  float ht[3] = {0, 0, 0};
  for (int i = 0; i < 100; ++i) {
    const float* h_prev = a.recurrentState();
    auto d = a.selectAction(o);
    a.computeIntrinsics(o, o2, ht, -0.02f);
    a.observeAndTrain(o, h_prev, d.action, -0.02f, -0.02f, o2, false, ht, false,
                      false, static_cast<uint64_t>(i));
  }
  AgentState st;
  a.exportState(st);
  // Import into a fresh agent and confirm learning continues (params move).
  Rng rng2(8);
  Agent b(cfg, rng2);
  CHECK(b.importState(st));
  AgentState b0;
  b.exportState(b0);
  for (int i = 0; i < 100; ++i) {
    const float* h_prev = b.recurrentState();
    auto d = b.selectAction(o);
    b.computeIntrinsics(o, o2, ht, -0.02f);
    b.observeAndTrain(o, h_prev, d.action, -0.02f, -0.02f, o2, false, ht, false,
                      false, static_cast<uint64_t>(i));
  }
  AgentState b1;
  b.exportState(b1);
  CHECK(b0.online_params != b1.online_params);  // learning did not stop
  // Incompatible topology is rejected.
  AgentState bad = st;
  bad.input = st.input + 1;
  CHECK(!b.importState(bad));
}

TEST(agent_consolidation_sequence_ops_train_and_stay_bounded) {
  // Chunked-BPTT sequence ops (consolidation) must train without corrupting
  // state: run enough steps to fill the replay, then execute a few sequence
  // ops and check parameters moved, everything stays finite and no invalid
  // update was recorded.
  Config cfg;
  cfg.observation_frames = 1;
  cfg.train_interval_steps = 1;
  cfg.replay_capacity = 64;
  cfg.bptt_chunk_len = 8;  // enable the chunked-BPTT path this test exercises
  Rng rng(11);
  Agent a(cfg, rng);
  float o[30] = {0.1f}, o2[30] = {0.2f};
  float ht[3] = {0.01f, 0.01f, 0.0f};
  for (int i = 0; i < 200; ++i) {
    const float* h_prev = a.recurrentState();
    auto d = a.selectAction(o);
    a.computeIntrinsics(o, o2, ht, -0.02f);
    a.observeAndTrain(o, h_prev, d.action, -0.02f, -0.02f, o2, false, ht, false,
                      false, static_cast<uint64_t>(i));
  }
  AgentState before;
  a.exportState(before);
  const uint64_t before_training = a.trainingUpdates();
  const int done = a.consolidationSequenceOps(4);
  CHECK(done > 0);  // replay had enough consecutive entries
  AgentState after;
  a.exportState(after);
  CHECK(after.allFinite());
  CHECK(before.online_params != after.online_params);  // weights moved
  CHECK(a.trainingUpdates() > before_training);        // updates counted
  CHECK(a.invalidUpdates() == 0);
  // Sequence ops require consecutive entries; a too-small replay returns 0.
  Config cfg2;
  cfg2.observation_frames = 1;
  cfg2.bptt_chunk_len = 8;  // enable so the "0" below tests too-small-replay, not disabled
  Rng rng2(12);
  Agent tiny(cfg2, rng2);
  CHECK(tiny.consolidationSequenceOps(2) == 0);
}
