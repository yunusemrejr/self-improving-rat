// Long-run boundedness and integration tests: memory stays at capacity,
// internal state stays in bounds, learning produces weight changes, and a
// headless agent+simulation loop remains stable over a substantial number of
// steps.

#include "test_framework.h"

#include "learning/agent.h"
#include "simulation/simulation.h"
#include "utility/config.h"
#include "utility/rng.h"

#include <cmath>
#include <vector>

using namespace sir;

namespace {

// Replicates Application::doSimStep (with the aliasing fix: pre-action
// observations and recurrent state are copied before mutating calls).
struct SimLoop {
  Config cfg;
  Rng rng;
  Simulation sim;
  Agent agent;
  explicit SimLoop(uint32_t seed)
      : cfg(), rng(seed), sim(cfg, rng), agent(cfg, rng) {}
  void step() {
    std::vector<float> s_local(agent.inputSize());
    std::copy(sim.observe(), sim.observe() + agent.inputSize(), s_local.begin());
    std::vector<float> h_prev_local(agent.rnnSize());
    std::copy(agent.recurrentState(), agent.recurrentState() + agent.rnnSize(),
              h_prev_local.begin());
    const float* s = s_local.data();
    const float* h_prev = h_prev_local.data();
    const Agent::Decision d = agent.selectAction(s);
    const Simulation::StepOutcome out = sim.step(d.action);
    const float* s2 = sim.observe();
    const auto& ha = out.homeo_after;
    const auto& hb = out.homeo_before;
    const float ht[3] = {ha.energy - hb.energy, ha.hunger - hb.hunger,
                         ha.stress - hb.stress};
    const double homeo_reward =
        -cfg.homeo_reward_energy * (1.0 - ha.energy) -
        cfg.homeo_reward_hunger * ha.hunger -
        cfg.homeo_reward_fatigue * ha.fatigue -
        cfg.homeo_reward_stress * ha.stress * ha.stress +
        cfg.homeo_reward_satisfaction * (ha.satisfaction - 0.5);
    const float reward_ext = out.reward_nav + static_cast<float>(homeo_reward);
    agent.computeIntrinsics(s, s2, ht, reward_ext);
    const double curiosity =
        static_cast<double>(agent.lastCuriosityReward()) *
        (0.5 + 0.5 * ha.curiosity_need);
    const double pred_bonus =
        cfg.prediction_reward_gain * (1.0 - agent.uncertainty());
    const int lf = (cfg.observation_frames - 1) * kObservationBase;
    const float scent_after = std::max(
        std::max(s2[lf + 8], s2[lf + 9]), std::max(s2[lf + 10], s2[lf + 11]));
    const double scent_proximity =
        cfg.scent_proximity_reward_gain * static_cast<double>(scent_after);
    const float r_total = static_cast<float>(
        std::max(-12.0, std::min(12.0,
            static_cast<double>(reward_ext) + curiosity + pred_bonus +
                scent_proximity)));
    agent.observeAndTrain(s, h_prev, d.action, r_total, reward_ext, s2,
                          out.cheese_reached, ht, out.cheese_reached,
                          out.wall_hit, sim.lifetimeSteps());
    sim.setNoveltySignal(agent.lastNovelty());
    sim.homeostasis().applyNovelty(agent.lastNovelty());
    sim.homeostasis().setUncertainty(agent.uncertainty());
  }
};

}  // namespace

TEST(longrun_bounded_memory_and_state) {
  SimLoop loop(99);
  const size_t episodic_cap = static_cast<size_t>(loop.cfg.episodic_memory_capacity);
  int cheese = 0;
  for (int i = 0; i < 20000; ++i) {
    loop.step();
    // Episodic memory is strictly bounded.
    CHECK(loop.agent.episodicSize() <= episodic_cap);
    CHECK(loop.agent.episodicCapacity() == episodic_cap);
    // Internal state stays in [0,1].
    const auto& h = loop.sim.homeostasis();
    CHECK(h.energy() >= 0.0f && h.energy() <= 1.0f);
    CHECK(h.hunger() >= 0.0f && h.hunger() <= 1.0f);
    CHECK(h.fatigue() >= 0.0f && h.fatigue() <= 1.0f);
    CHECK(h.stress() >= 0.0f && h.stress() <= 1.0f);
    if (i == 10000) {
      // No unbounded growth at the halfway point.
      CHECK(loop.agent.episodicSize() <= episodic_cap);
    }
    if (i > 0 && i % 10000 == 0) {
      // Everything is still finite and stable.
      AgentState st;
      loop.agent.exportState(st);
      CHECK(st.allFinite());
    }
    if (i % 1000 == 0) {
      // Replay has no public size() on Agent; verify indirectly that
      // training still produces updates (buffer stays functional).
      const uint64_t expected = i >= 32 ? static_cast<uint64_t>(i) / 2 - 100 : 0;
      CHECK(loop.agent.trainingUpdates() >= expected);
    }
    if (loop.sim.cheeseTotal() > static_cast<uint64_t>(cheese)) {
      cheese = static_cast<int>(loop.sim.cheeseTotal());
    }
  }
  (void)cheese;
}

TEST(learning_weights_change_over_long_loop) {
  SimLoop loop(7);
  // Snapshot initial weights.
  AgentState s0;
  loop.agent.exportState(s0);
  for (int i = 0; i < 3000; ++i) loop.step();
  AgentState s1;
  loop.agent.exportState(s1);
  CHECK(s0.online_params != s1.online_params);
  CHECK(loop.agent.trainingUpdates() > 0);
  CHECK(loop.agent.invalidUpdates() == 0);
  // Recurrent state and prediction signals are all finite.
  CHECK(loop.agent.uncertainty() >= 0.0f && loop.agent.uncertainty() <= 1.0f);
  CHECK(loop.agent.lastPredError() >= 0.0f && loop.agent.lastPredError() <= 1.0f);
}

TEST(learning_integration_finds_some_cheese_across_seeds) {
  // The default 13x9 maze must be navigable: across 4 seeds, the agent finds
  // cheese at least once in total (the life loop demonstrably runs).
  int cheese_total = 0;
  for (uint32_t seed = 1; seed <= 4; ++seed) {
    SimLoop loop(seed);
    for (int i = 0; i < 15000; ++i) loop.step();
    cheese_total += static_cast<int>(loop.sim.cheeseTotal());
  }
  CHECK(cheese_total >= 1);
}

TEST(replay_capacity_respected_end_to_end) {
  Config cfg;
  cfg.observation_frames = 1;
  cfg.replay_capacity = 128;
  cfg.train_interval_steps = 1;
  Rng rng(5);
  Agent a(cfg, rng);
  float o[30] = {0.1f}, o2[30] = {0.2f};
  float ht[3] = {0, 0, 0};
  for (int i = 0; i < 10000; ++i) {
    const float* h_prev = a.recurrentState();
    auto d = a.selectAction(o);
    a.computeIntrinsics(o, o2, ht, -0.02f);
    a.observeAndTrain(o, h_prev, d.action, -0.02f, -0.02f, o2, false, ht,
                      false, false, static_cast<uint64_t>(i));
  }
  CHECK(a.trainingUpdates() > 5000);  // training never stalls
  CHECK(a.invalidUpdates() == 0);
  AgentState st;
  a.exportState(st);
  CHECK(st.allFinite());
}

TEST(deterministic_seed_identical_agent_trajectory) {
  // Two runs with the same seed produce identical action sequences and
  // identical serialized parameters at the same step count (no hidden
  // nondeterminism; the checkpoint timestamps are excluded by comparing
  // parameters only).
  auto run = [](uint32_t seed) {
    SimLoop loop(seed);
    std::vector<uint32_t> actions;
    for (int i = 0; i < 300; ++i) {
      // Capture the greedy action derived from qValuesFor (no RNG draws, so
      // the probe is deterministic).
      std::vector<float> q(4);
      std::vector<float> obs(loop.agent.inputSize());
      std::copy(loop.sim.observe(), loop.sim.observe() + loop.agent.inputSize(),
                obs.begin());
      loop.agent.qValuesFor(obs.data(), q.data());
      int best = 0;
      for (int k = 1; k < 4; ++k)
        if (q[k] > q[best]) best = k;
      actions.push_back(static_cast<uint32_t>(best));
      loop.step();
    }
    AgentState st;
    loop.agent.exportState(st);
    return std::make_pair(actions, st.online_params);
  };
  const auto r1 = run(777);
  const auto r2 = run(777);
  CHECK(r1.first == r2.first);
  CHECK(r1.second == r2.second);
  const auto r3 = run(778);
  CHECK(r1.first != r3.first);  // different seed diverges
}
