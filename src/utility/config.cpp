#include "utility/config.h"
#include <algorithm>

#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>

namespace sir {

namespace {

// Reads one "key = value" line. Returns false for blank/comment lines.
bool parseLine(const std::string& line, std::string* key, std::string* value) {
  std::string s = line;
  size_t comment = s.find('#');
  if (comment != std::string::npos) s = s.substr(0, comment);
  size_t eq = s.find('=');
  if (eq == std::string::npos) return false;
  *key = s.substr(0, eq);
  *value = s.substr(eq + 1);
  // trim
  auto trim = [](std::string& t) {
    size_t b = t.find_first_not_of(" \t\r\n");
    size_t e = t.find_last_not_of(" \t\r\n");
    if (b == std::string::npos) { t.clear(); return; }
    t = t.substr(b, e - b + 1);
  };
  trim(*key);
  trim(*value);
  return !key->empty();
}

}  // namespace

int Config::loadFromFile(const std::string& path, std::string* error,
                         std::string* warnings) {
  std::ifstream in(path);
  if (!in.is_open()) {
    if (error) *error = "cannot open config file: " + path;
    return -1;
  }
  int applied = 0;
  std::string line;
  int line_no = 0;
  auto warn = [&](const std::string& m) {
    if (warnings) *warnings += "  line " + std::to_string(line_no) + ": " + m + "\n";
  };
  while (std::getline(in, line)) {
    ++line_no;
    std::string k, v;
    if (!parseLine(line, &k, &v)) continue;

    auto setInt = [&](int* dst, int lo, int hi) {
      try {
        int val = std::stoi(v);
        *dst = std::max(lo, std::min(hi, val));
        ++applied;
      } catch (...) { warn("invalid integer for '" + k + "': " + v); }
    };
    auto setI64 = [&](int64_t* dst, int64_t lo, int64_t hi) {
      try {
        long long val = std::stoll(v);
        *dst = std::max(lo, std::min(hi, static_cast<int64_t>(val)));
        ++applied;
      } catch (...) { warn("invalid integer for '" + k + "': " + v); }
    };
    auto setU32 = [&](uint32_t* dst) {
      try {
        long long val = std::stoll(v);
        if (val < 0) { warn("negative value for '" + k + "'"); return; }
        *dst = static_cast<uint32_t>(val);
        ++applied;
      } catch (...) { warn("invalid value for '" + k + "': " + v); }
    };
    auto setU64 = [&](uint64_t* dst) {
      try {
        long long val = std::stoll(v);
        if (val < 0) { warn("negative value for '" + k + "'"); return; }
        *dst = static_cast<uint64_t>(val);
        ++applied;
      } catch (...) { warn("invalid value for '" + k + "': " + v); }
    };
    auto setD = [&](double* dst) {
      try {
        *dst = std::stod(v);
        if (!std::isfinite(*dst)) { warn("non-finite value for '" + k + "'"); return; }
        ++applied;
      } catch (...) { warn("invalid number for '" + k + "': " + v); }
    };
    auto setS = [&](std::string* dst) { *dst = v; ++applied; };
    auto setB = [&](bool* dst) {
      std::string low;
      for (char c : v) low += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      if (low == "true" || low == "1" || low == "yes" || low == "on") *dst = true;
      else if (low == "false" || low == "0" || low == "no" || low == "off") *dst = false;
      else { warn("invalid boolean for '" + k + "': " + v); return; }
      ++applied;
    };

    if (k == "window_width") setInt(&window_width, 320, 4096);
    else if (k == "window_height") setInt(&window_height, 240, 4096);
    else if (k == "maze_width") setInt(&maze_width, 5, 199);
    else if (k == "maze_height") setInt(&maze_height, 5, 199);
    else if (k == "tile_size") setInt(&tile_size, 4, 64);
    else if (k == "render_frames_per_second") setInt(&render_frames_per_second, 1, 240);
    else if (k == "debug_display") setB(&debug_display);
    else if (k == "sim_steps_per_second") setInt(&sim_steps_per_second, 1, 1000);
    else if (k == "maze_braid_probability") {
      setD(&maze_braid_probability);
      maze_braid_probability = std::max(0.0, std::min(1.0, maze_braid_probability));
    } else if (k == "maze_regenerate_every_cheeses") setInt(&maze_regenerate_every_cheeses, 1, 100000);
    else if (k == "min_cheese_distance") setInt(&min_cheese_distance, 2, 200);
    else if (k == "sensory_radius") setInt(&sensory_radius, 1, 30);
    else if (k == "revisit_window_steps") setInt(&revisit_window_steps, 1, 32);
    else if (k == "reward_cheese") setD(&reward_cheese);
    else if (k == "reward_wall_hit") setD(&reward_wall_hit);
    else if (k == "reward_step") setD(&reward_step);
    else if (k == "reward_revisit") setD(&reward_revisit);
    else if (k == "nn_hidden") setInt(&nn_hidden, 4, 512);
    else if (k == "rnn_hidden") setInt(&rnn_hidden, 4, 128);
    else if (k == "observation_frames") setInt(&observation_frames, 1, 4);
    else if (k == "learning_rate") {
      setD(&learning_rate);
      learning_rate = std::max(0.0, std::min(0.1, learning_rate));
    } else if (k == "discount_factor") {
      setD(&discount_factor);
      discount_factor = std::max(0.0, std::min(1.0, discount_factor));
    } else if (k == "target_update_tau") {
      setD(&target_update_tau);
      target_update_tau = std::max(0.0, std::min(1.0, target_update_tau));
    } else if (k == "exploration_start") {
      setD(&exploration_start);
      exploration_start = std::max(0.0, std::min(1.0, exploration_start));
    } else if (k == "exploration_end") {
      setD(&exploration_end);
      exploration_end = std::max(0.0, std::min(1.0, exploration_end));
    } else if (k == "exploration_decay_steps") setInt(&exploration_decay_steps, 0, 100000000);
    else if (k == "replay_capacity") setInt(&replay_capacity, 64, 1 << 20);
    else if (k == "batch_size") setInt(&batch_size, 1, 1024);
    else if (k == "train_interval_steps") setInt(&train_interval_steps, 1, 1024);
    else if (k == "gradient_clip_norm") {
      setD(&gradient_clip_norm);
      gradient_clip_norm = std::max(1e-6, gradient_clip_norm);
    } else if (k == "n_step_returns") setInt(&n_step_returns, 1, 8);
    else if (k == "td_huber_delta") {
      setD(&td_huber_delta);
      td_huber_delta = std::max(0.01, std::min(100.0, td_huber_delta));
    } else if (k == "per_is_beta") {
      setD(&per_is_beta);
      per_is_beta = std::max(0.0, std::min(1.0, per_is_beta));
    } else if (k == "per_priority_alpha") {
      setD(&per_priority_alpha);
      per_priority_alpha = std::clamp(per_priority_alpha, 0.0, 1.0);
    } else if (k == "episodic_action_bonus") {
      setD(&episodic_action_bonus);
      episodic_action_bonus = std::clamp(episodic_action_bonus, 0.0, 2.0);
    } else if (k == "mask_wall_actions") setB(&mask_wall_actions);
    else if (k == "replay_burn_in") setInt(&replay_burn_in, 0, 16);
    else if (k == "sequence_train_interval") setInt(&sequence_train_interval, 0, 4096);
    else if (k == "bptt_chunk_len") setInt(&bptt_chunk_len, 0, 8);
    // --- organism: homeostasis ---
    else if (k == "energy_cost_step") setD(&energy_cost_step);
    else if (k == "energy_cost_move") setD(&energy_cost_move);
    else if (k == "energy_cheese_gain") setD(&energy_cheese_gain);
    else if (k == "hunger_rate") setD(&hunger_rate);
    else if (k == "hunger_cheese_reduction") setD(&hunger_cheese_reduction);
    else if (k == "fatigue_per_move") setD(&fatigue_per_move);
    else if (k == "fatigue_recovery_rest") setD(&fatigue_recovery_rest);
    else if (k == "fatigue_recovery_idle") setD(&fatigue_recovery_idle);
    else if (k == "stress_wall_hit") setD(&stress_wall_hit);
    else if (k == "stress_repeat_bonus") setD(&stress_repeat_bonus);
    else if (k == "stress_success_relief") setD(&stress_success_relief);
    else if (k == "stress_cheese_relief") setD(&stress_cheese_relief);
    else if (k == "stress_recovery_rate") setD(&stress_recovery_rate);
    else if (k == "curiosity_need_rate") setD(&curiosity_need_rate);
    else if (k == "curiosity_need_reduce") setD(&curiosity_need_reduce);
    else if (k == "satisfaction_cheese") setD(&satisfaction_cheese);
    else if (k == "satisfaction_reversion") setD(&satisfaction_reversion);
    // --- organism: reward composition ---
    else if (k == "homeo_reward_energy") setD(&homeo_reward_energy);
    else if (k == "homeo_reward_hunger") setD(&homeo_reward_hunger);
    else if (k == "homeo_reward_fatigue") setD(&homeo_reward_fatigue);
    else if (k == "homeo_reward_stress") setD(&homeo_reward_stress);
    else if (k == "homeo_reward_satisfaction") setD(&homeo_reward_satisfaction);
    else if (k == "curiosity_reward_gain") setD(&curiosity_reward_gain);
    else if (k == "prediction_reward_gain") setD(&prediction_reward_gain);
    else if (k == "collapse_penalty") setD(&collapse_penalty);
    else if (k == "scent_proximity_reward_gain") {
      setD(&scent_proximity_reward_gain);
      scent_proximity_reward_gain = std::max(0.0, scent_proximity_reward_gain);
    }
    // --- organism: prediction / curiosity ---
    else if (k == "prediction_loss_weight") setD(&prediction_loss_weight);
    else if (k == "prediction_ema_decay") {
      setD(&prediction_ema_decay);
      prediction_ema_decay = std::max(0.0, std::min(1.0, prediction_ema_decay));
    } else if (k == "novelty_state_weight") {
      setD(&novelty_state_weight);
      novelty_state_weight = std::max(0.0, std::min(1.0, novelty_state_weight));
    } else if (k == "novelty_pred_weight") {
      setD(&novelty_pred_weight);
      novelty_pred_weight = std::max(0.0, std::min(1.0, novelty_pred_weight));
    } else if (k == "novelty_table_capacity") setInt(&novelty_table_capacity, 64, 1 << 16);
    // --- organism: episodic memory ---
    else if (k == "episodic_memory_capacity") setInt(&episodic_memory_capacity, 8, 1 << 14);
    // --- organism: development ---
    else if (k == "maturity_steps") setU64(&maturity_steps);
    else if (k == "lr_age_scale") {
      setD(&lr_age_scale);
      lr_age_scale = std::max(0.0, std::min(1.0, lr_age_scale));
    } else if (k == "epsilon_age_scale") {
      setD(&epsilon_age_scale);
      epsilon_age_scale = std::max(0.0, std::min(1.0, epsilon_age_scale));
    } else if (k == "plasticity_age_scale") {
      setD(&plasticity_age_scale);
      plasticity_age_scale = std::max(0.0, std::min(1.0, plasticity_age_scale));
    }
    // --- organism: consolidation ---
    else if (k == "consolidation_every_cheeses") setInt(&consolidation_every_cheeses, 1, 100000);
    else if (k == "consolidation_min_interval_steps") setU64(&consolidation_min_interval_steps);
    else if (k == "consolidation_max_steps") setInt(&consolidation_max_steps, 1, 100000);
    else if (k == "consolidation_max_train_ops") setInt(&consolidation_max_train_ops, 1, 100000);
    else if (k == "consolidation_fatigue_threshold") {
      setD(&consolidation_fatigue_threshold);
      consolidation_fatigue_threshold = std::max(0.0, std::min(1.0, consolidation_fatigue_threshold));
    }
    // --- organism: structural plasticity ---
    else if (k == "plasticity_max_active_fraction") {
      setD(&plasticity_max_active_fraction);
      plasticity_max_active_fraction = std::max(0.1, std::min(1.0, plasticity_max_active_fraction));
    } else if (k == "plasticity_rewire_per_consolidation") setInt(&plasticity_rewire_per_consolidation, 0, 10000);
    else if (k == "plasticity_utility_decay") {
      setD(&plasticity_utility_decay);
      plasticity_utility_decay = std::max(0.0, std::min(1.0, plasticity_utility_decay));
    } else if (k == "plasticity_dormant_threshold_fraction") {
      setD(&plasticity_dormant_threshold_fraction);
      plasticity_dormant_threshold_fraction = std::max(0.0, std::min(0.5, plasticity_dormant_threshold_fraction));
    } else if (k == "plasticity_strengthen_factor") {
      setD(&plasticity_strengthen_factor);
      plasticity_strengthen_factor = std::max(0.0, std::min(1.0, plasticity_strengthen_factor));
    } else if (k == "plasticity_eval_passes") setInt(&plasticity_eval_passes, 1, 10000);
    else if (k == "autosave_interval_seconds") setInt(&autosave_interval_seconds, 1, 86400);
    else if (k == "checkpoint_dir") setS(&checkpoint_dir);
    else if (k == "log_file") setS(&log_file);
    else if (k == "log_max_bytes") setI64(&log_max_bytes, 4096, 1LL << 30);
    else if (k == "max_catchup_steps_per_frame") setInt(&max_catchup_steps_per_frame, 1, 100000);
    else if (k == "seed") setU32(&seed);
    else warn("unknown key '" + k + "'");
  }
  return applied;
}

}  // namespace sir
