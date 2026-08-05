#pragma once
// Application orchestration: main loop, pacing, controls, consolidation,
// autosave, checkpoint recovery, headless/diagnostic hooks. Single-threaded.

#include "learning/agent.h"
#include "persistence/checkpoint.h"
#include "rendering/renderer.h"
#include "simulation/simulation.h"
#include "utility/config.h"
#include "utility/logger.h"
#include "utility/rng.h"
#include "utility/rolling_stats.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sir {

// Incremented by signal handlers (main.cpp); polled by the main loop.
// Second signal forces immediate exit.
extern std::atomic<int> g_signal_count;

class Application {
 public:
  struct Options {
    std::string config_path;  // empty = config/default.cfg
    bool headless = false;
    std::string screenshot_path;  // empty = disabled
    uint64_t max_steps = 0;       // 0 = unlimited
    uint32_t seed_override = 0;   // non-zero overrides the config seed (SIR_SEED)
  };

  int run(const Options& opts);

 private:
  void saveCheckpointNow(const char* reason);
  void loadCheckpoint();
  void doSimStep();
  void doConsolidationStep();
  bool shouldConsolidate() const;
  void beginConsolidation();
  void endConsolidation();
  void fillPanel(PanelData& p);
  void pollSignals();
  uint64_t nowMs() const;

  Config cfg_;
  Logger log_;
  std::unique_ptr<Rng> rng_;
  std::unique_ptr<Simulation> sim_;
  std::unique_ptr<Agent> agent_;
  std::unique_ptr<CheckpointStore> store_;
  Metrics metrics_;
  Renderer renderer_;

  bool paused_ = false;
  bool debug_ = false;
  bool consolidating_ = false;
  int consolidation_steps_left_ = 0;
  int consolidation_ops_left_ = 0;
  uint64_t consolidation_cycles_ = 0;
  uint64_t last_consolidation_step_ = 0;
  int cheese_since_last_consolidation_ = 0;
  uint64_t session_start_ms_ = 0;
  uint64_t loaded_runtime_ms_ = 0;
  uint64_t last_autosave_ms_ = 0;
  uint64_t checkpoints_saved_ = 0;
  uint64_t last_sim_step_ms_ = 0;
  double sim_accum_ = 0.0;
  int prev_action_ = -1;
  int adaptation_remaining_ = -1;  // steps since last maze regen (starts -1)
  uint64_t last_fps_ms_ = 0;
  int frames_since_fps_ = 0;
  double fps_ = 0.0;
  double sim_sps_ = 0.0;
  uint64_t last_sps_ms_ = 0;
  int steps_since_sps_ = 0;
  std::vector<float> recent_obs_;  // ring for plasticity evaluation
  int recent_obs_head_ = 0;
  bool screenshot_done_ = false;
  int frames_rendered_ = 0;
};

}  // namespace sir
