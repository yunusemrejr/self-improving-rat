#pragma once
// Central configuration. Built-in defaults match config/default.cfg shipped
// with the project; a missing or partial config file falls back to them.
// All numeric values are clamped to safe ranges on load (see config.cpp).

#include <cstdint>
#include <string>

namespace sir {

struct Config {
  // Window / rendering
  int window_width = 960;
  int window_height = 640;
  int maze_width = 13;
  int maze_height = 9;
  int tile_size = 24;
  int render_frames_per_second = 30;
  bool debug_display = false;

  // Simulation
  int sim_steps_per_second = 12;
  double maze_braid_probability = 0.4;
  int maze_regenerate_every_cheeses = 6;
  int min_cheese_distance = 3;
  int sensory_radius = 20;   // cheese scent range in cells (covers the maze)
  int revisit_window_steps = 4;

  // Navigation rewards (fixed base)
  double reward_cheese = 10.0;
  double reward_wall_hit = -0.2;
  double reward_step = -0.02;
  double reward_revisit = -0.05;

  // Learning
  int nn_hidden = 32;  // unused legacy knob; kept for compatibility
  int rnn_hidden = 16;
  int observation_frames = 1;  // stacked base observations; input = 28 * frames
  double learning_rate = 0.0003;
  double discount_factor = 0.95;
  double target_update_tau = 0.02;
  double exploration_start = 0.9;
  double exploration_end = 0.05;
  int exploration_decay_steps = 50000;
  int replay_capacity = 4096;
  int batch_size = 16;
  int train_interval_steps = 2;
  double gradient_clip_norm = 1.0;

  // --- Organism: homeostasis (see organism/homeostasis.h) ---
  double energy_cost_step = 0.0004;
  double energy_cost_move = 0.0008;
  double energy_cheese_gain = 0.35;
  double hunger_rate = 0.0003;
  double hunger_cheese_reduction = 0.5;
  double fatigue_per_move = 0.0006;
  double fatigue_recovery_rest = 0.004;
  double fatigue_recovery_idle = 0.0002;
  double stress_wall_hit = 0.04;
  double stress_repeat_bonus = 0.03;
  double stress_success_relief = 0.002;
  double stress_cheese_relief = 0.2;
  double stress_recovery_rate = 0.0005;
  double curiosity_need_rate = 0.0002;
  double curiosity_need_reduce = 0.02;
  double satisfaction_cheese = 0.4;
  double satisfaction_reversion = 0.001;

  // --- Organism: composite reward composition ---
  double homeo_reward_energy = 0.002;
  double homeo_reward_hunger = 0.004;
  double homeo_reward_fatigue = 0.002;
  double homeo_reward_stress = 0.004;
  double homeo_reward_satisfaction = 0.001;
  double curiosity_reward_gain = 0.02;
  double prediction_reward_gain = 0.01;
  double collapse_penalty = 0.01;
  // Dense shaping from the rat's own scent perception: a bonus proportional
  // to the strongest scent channel after the action (rewarding being near the
  // cheese). Unlike a delta form it does not punish the detours a maze
  // requires, so the value function develops a gradient toward the cheese
  // that transfers across cheese re-placements.
  double scent_proximity_reward_gain = 0.08;

  // --- Organism: prediction / curiosity ---
  double prediction_loss_weight = 0.3;
  double prediction_ema_decay = 0.95;  // uncertainty EMA decay per training step
  double novelty_state_weight = 0.5;   // novelty blend: state vs prediction error
  double novelty_pred_weight = 0.5;
  int novelty_table_capacity = 1024;

  // --- Organism: episodic memory ---
  int episodic_memory_capacity = 256;

  // --- Organism: lifelong development ---
  uint64_t maturity_steps = 200000;
  double lr_age_scale = 0.5;
  double epsilon_age_scale = 0.3;
  double plasticity_age_scale = 0.7;

  // --- Organism: consolidation ---
  int consolidation_every_cheeses = 4;
  uint64_t consolidation_min_interval_steps = 2000;
  int consolidation_max_steps = 120;
  int consolidation_max_train_ops = 60;
  double consolidation_fatigue_threshold = 0.9;

  // --- Organism: structural plasticity ---
  double plasticity_max_active_fraction = 0.6;
  int plasticity_rewire_per_consolidation = 24;
  double plasticity_utility_decay = 0.999;
  double plasticity_dormant_threshold_fraction = 0.05;
  double plasticity_strengthen_factor = 0.02;
  int plasticity_eval_passes = 32;

  // Persistence / logging
  int autosave_interval_seconds = 15;
  std::string checkpoint_dir = "data/checkpoints";
  std::string log_file = "data/logs/rat.log";
  int64_t log_max_bytes = 524288;  // 512 KiB, then rotate

  // Resource limits
  int max_catchup_steps_per_frame = 8;

  // Randomness: 0 = random seed at startup; non-zero = reproducible.
  uint32_t seed = 0;

  // Loads "key = value" lines from a file. Returns the number of keys applied.
  // On failure returns -1 and sets *error. Invalid lines are skipped with a
  // warning appended to *warnings (may be null).
  int loadFromFile(const std::string& path, std::string* error,
                   std::string* warnings);
};

}  // namespace sir
