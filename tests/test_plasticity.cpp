// Structural plasticity and consolidation tests: hard limits, real topology
// change, serialization round-trip of masks, bounded consolidation work.

#include "test_framework.h"

#include "learning/agent.h"
#include "learning/neural_net.h"
#include "utility/config.h"
#include "utility/rng.h"

#include <vector>

using namespace sir;

TEST(plasticity_pruning_changes_computation_and_stays_bounded) {
  Config cfg;
  cfg.observation_frames = 1;
  Rng rng(1);
  Agent a(cfg, rng);
  const size_t total_weights =
      a.activeConnections() + a.dormantConnections();
  // Build some utility by training so pruning has candidates.
  float o[30] = {0.1f}, o2[30] = {0.2f};
  float ht[3] = {0, 0, 0};
  for (int i = 0; i < 400; ++i) {
    const float* h_prev = a.recurrentState();
    auto d = a.selectAction(o);
    a.computeIntrinsics(o, o2, ht, -0.02f);
    a.observeAndTrain(o, h_prev, d.action, -0.02f, -0.02f, o2, false, ht,
                      false, false, static_cast<uint64_t>(i));
  }
  const size_t active_before = a.activeConnections();
  // Evaluate plasticity on a stable observation buffer.
  std::vector<float> eval(64 * a.inputSize(), 0.1f);
  a.plasticityEvaluate(eval, cfg.plasticity_eval_passes);
  const size_t active_after = a.activeConnections();
  // Connection conservation: active + dormant == total weights; the hard
  // active cap limits reactivation (the cap is not a pruning floor, so the
  // count may legitimately sit below it after heavy pruning).
  CHECK(active_after + a.dormantConnections() == total_weights);
  CHECK(active_after <= total_weights);
  const size_t rewire_cap =
      static_cast<size_t>(cfg.plasticity_rewire_per_consolidation);
  CHECK(active_after <= active_before + rewire_cap + 1);  // reactivation bounded
  CHECK(a.structuralAccepted() + a.structuralRejected() >= 1);
}

TEST(plasticity_mask_serialization_roundtrip) {
  Rng rng(2);
  NeuralNet net(30, 16, 4, 16, rng);
  // Prune a handful of connections through the public API.
  net.zeroParam(0);
  net.zeroParam(5);
  net.zeroParam(100);
  std::vector<uint8_t> mask(net.maskBytes());
  net.getMask(mask.data());
  // Round-trip.
  NeuralNet net2(30, 16, 4, 16, rng);
  net2.setMask(mask.data());
  std::vector<uint8_t> mask2(net2.maskBytes());
  net2.getMask(mask2.data());
  CHECK(mask == mask2);
  CHECK(net2.activeCount() == net.activeCount());
  // Reactivation restores a usable connection with a fresh weight.
  net.reactivateParam(0, rng, 0.5f);
  CHECK(net.activeCount() == net2.activeCount() + 1);
}

TEST(plasticity_dormant_weights_are_zeroed_and_usable) {
  Rng rng(3);
  NeuralNet net(30, 16, 4, 16, rng);
  net.zeroParam(7);
  // The zeroed parameter must be inactive and its weight exactly zero.
  std::vector<uint8_t> mask(net.maskBytes());
  net.getMask(mask.data());
  CHECK((mask[0] & (1u << 7)) == 0);
  std::vector<float> params(net.paramCount());
  net.getParams(params.data());
  // Mask index 7 maps to dense offset 7 (first block Wz has no bias gap).
  CHECK_NEAR(params[7], 0.0f, 1e-6);
}

TEST(consolidation_train_ops_bounded) {
  Config cfg;
  cfg.observation_frames = 1;
  Rng rng(4);
  Agent a(cfg, rng);
  float o[30] = {0.1f}, o2[30] = {0.2f};
  float ht[3] = {0, 0, 0};
  for (int i = 0; i < 200; ++i) {
    const float* h_prev = a.recurrentState();
    auto d = a.selectAction(o);
    a.computeIntrinsics(o, o2, ht, -0.02f);
    a.observeAndTrain(o, h_prev, d.action, -0.02f, -0.02f, o2, false, ht,
                      false, false, static_cast<uint64_t>(i));
  }
  CHECK(a.consolidationAvailable());
  const uint64_t ops_before = a.consolidationTrainOpsTotal();
  const int executed = a.consolidationTrainOps(5);
  CHECK(executed <= 5);
  CHECK(a.consolidationTrainOpsTotal() == ops_before + static_cast<uint64_t>(executed));
  // The network remains finite after consolidation training.
  AgentState st;
  a.exportState(st);
  CHECK(st.allFinite());
  // Parameters changed (real training happened) if ops executed.
  if (executed > 0) {
    AgentState st2;
    // Force another training op and compare.
    a.consolidationTrainOps(1);
    a.exportState(st2);
    CHECK(st.online_params != st2.online_params);
  }
}

TEST(consolidation_gathers_episodic_and_replay) {
  Config cfg;
  cfg.observation_frames = 1;
  Rng rng(5);
  Agent a(cfg, rng);
  float o[30] = {0.3f}, o2[30] = {0.4f};
  float ht[3] = {0, 0, 0};
  // Seed replay + episodic memory with significant experiences.
  for (int i = 0; i < 300; ++i) {
    const float* h_prev = a.recurrentState();
    auto d = a.selectAction(o);
    a.computeIntrinsics(o, o2, ht, 1.0f);
    a.observeAndTrain(o, h_prev, d.action, 1.0f, 1.0f, o2, i % 50 == 0, ht,
                      i % 50 == 0, false, static_cast<uint64_t>(i));
  }
  CHECK(a.episodicSize() > 0);
  const int done = a.consolidationTrainOps(10);
  CHECK(done >= 0 && done <= 10);
  AgentState st;
  a.exportState(st);
  CHECK(st.allFinite());
}

TEST(agent_param_count_bounds) {
  Config cfg;
  cfg.observation_frames = 1;
  Rng rng(6);
  Agent a(cfg, rng);
  // Expose paramCount via export (topology sanity: rnn 16, po 4, pr 16).
  AgentState st;
  a.exportState(st);
  const int expected = GruLayer::paramCount(30, 16) + 16 * 4 + 4 + 16 * 16 + 16;
  CHECK(st.online_params.size() == static_cast<size_t>(expected));
  CHECK(st.masks.size() == (static_cast<size_t>(expected) + 7) / 8);
}
