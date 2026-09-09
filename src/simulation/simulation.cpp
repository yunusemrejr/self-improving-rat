#include "simulation/simulation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sir {

namespace {
constexpr int kDirX[4] = {0, 0, -1, 1};
constexpr int kDirY[4] = {-1, 1, 0, 0};

HomeoParams homeoParamsFromConfig(const Config& cfg) {
  HomeoParams p;
  p.energy_cost_step = static_cast<float>(cfg.energy_cost_step);
  p.energy_cost_move = static_cast<float>(cfg.energy_cost_move);
  p.energy_cheese_gain = static_cast<float>(cfg.energy_cheese_gain);
  p.hunger_rate = static_cast<float>(cfg.hunger_rate);
  p.hunger_cheese_reduction = static_cast<float>(cfg.hunger_cheese_reduction);
  p.fatigue_per_move = static_cast<float>(cfg.fatigue_per_move);
  p.fatigue_recovery_rest = static_cast<float>(cfg.fatigue_recovery_rest);
  p.fatigue_recovery_idle = static_cast<float>(cfg.fatigue_recovery_idle);
  p.stress_wall_hit = static_cast<float>(cfg.stress_wall_hit);
  p.stress_repeat_bonus = static_cast<float>(cfg.stress_repeat_bonus);
  p.stress_success_relief = static_cast<float>(cfg.stress_success_relief);
  p.stress_cheese_relief = static_cast<float>(cfg.stress_cheese_relief);
  p.stress_recovery_rate = static_cast<float>(cfg.stress_recovery_rate);
  p.curiosity_need_rate = static_cast<float>(cfg.curiosity_need_rate);
  p.curiosity_need_reduce = static_cast<float>(cfg.curiosity_need_reduce);
  p.satisfaction_cheese = static_cast<float>(cfg.satisfaction_cheese);
  p.satisfaction_reversion = static_cast<float>(cfg.satisfaction_reversion);
  return p;
}
}  // namespace

Simulation::Simulation(const Config& cfg, Rng& rng)
    : cfg_(cfg),
      rng_(rng),
      maze_(cfg.maze_width, cfg.maze_height),
      homeo_(homeoParamsFromConfig(cfg)),
      last_visit_step_(static_cast<size_t>(cfg.maze_width) * cfg.maze_height, -1),
      input_size_(kObservationBase * cfg.observation_frames) {
  frame_history_.assign(static_cast<size_t>(input_size_), 0.0f);
  maze_.generate(rng_, cfg.maze_braid_probability);
  resetRatAndCheese();
  pushFrame();  // first observation (previous frames zero-filled)
}

void Simulation::resetRatAndCheese() {
  // Random walkable start for the rat.
  int rx = 0, ry = 0;
  do {
    rx = rng_.uniformInt(1, maze_.width() - 2);
    ry = rng_.uniformInt(1, maze_.height() - 2);
  } while (maze_.isWall(rx, ry));
  rat_.setPosition(rx, ry);
  rat_.setFacing(Action::Up);
  rat_.setLastAction(Action::Up);
  rat_.setWallHitLastAction(false);

  std::fill(last_visit_step_.begin(), last_visit_step_.end(), -1);
  last_visit_step_[static_cast<size_t>(ry) * maze_.width() + rx] = 0;

  placeCheese();
}

void Simulation::placeCheese() {
  cheeses_.clear();
  const int rx = rat_.position().x;
  const int ry = rat_.position().y;
  int cx = 1, cy = 1;
  bool placed = false;
  // Try up to 300 random walkable cells at minimum distance; fall back to the
  // farthest walkable cell (always possible in a connected maze).
  for (int attempt = 0; attempt < 300; ++attempt) {
    cx = rng_.uniformInt(1, maze_.width() - 2);
    cy = rng_.uniformInt(1, maze_.height() - 2);
    if (maze_.isWall(cx, cy)) continue;
    if (std::abs(cx - rx) + std::abs(cy - ry) >= cfg_.min_cheese_distance) {
      placed = true;
      break;
    }
  }
  if (!placed) {
    int best_d = -1;
    for (int y = 1; y < maze_.height() - 1; ++y) {
      for (int x = 1; x < maze_.width() - 1; ++x) {
        if (maze_.isWall(x, y)) continue;
        int d = std::abs(x - rx) + std::abs(y - ry);
        if (d > best_d) {
          best_d = d;
          cx = x;
          cy = y;
        }
      }
    }
  }
  cheeses_.push_back({cx, cy});
}

void Simulation::regenerateMaze() {
  maze_.generate(rng_, cfg_.maze_braid_probability);
  ++maze_generations_;
  steps_since_cheese_ = 0;
  resetRatAndCheese();
  pushFrame();
}

void Simulation::pushFrame() {
  float base[kObservationBase];
  const ObservationContext ctx{
      homeo_.energy(),         homeo_.hunger(),       homeo_.fatigue(),
      homeo_.stress(),         homeo_.curiosityNeed(), homeo_.satisfaction(),
      homeo_.uncertainty(),    novelty_signal_,        resting_ ? 1.0f : 0.0f};
  buildObservation(maze_, rat_.position().x, rat_.position().y,
                   rat_.lastAction(), rat_.wallHitLastAction(),
                   last_revisit_signal_, last_reward_, cheeses_,
                   cfg_.sensory_radius, ctx, base);
  if (first_frame_) {
    // Fill the whole history with the first observation (no prior frames).
    for (int i = 0; i < cfg_.observation_frames; ++i) {
      std::copy(base, base + kObservationBase,
                frame_history_.data() + static_cast<size_t>(i) * kObservationBase);
    }
    first_frame_ = false;
  } else {
    // Shift older frames down and append the newest at the end.
    std::copy(frame_history_.begin() + kObservationBase,
              frame_history_.end(), frame_history_.begin());
    std::copy(base, base + kObservationBase,
              frame_history_.end() - kObservationBase);
  }
}

Simulation::StepOutcome Simulation::step(Action a) {
  StepOutcome out;
  out.homeo_before = homeo_.snapshot();

  const int dx = kDirX[static_cast<int>(a)];
  const int dy = kDirY[static_cast<int>(a)];
  const Position& p = rat_.position();
  const int nx = p.x + dx, ny = p.y + dy;

  rat_.setLastAction(a);
  rat_.setFacing(a);

  float reward = static_cast<float>(cfg_.reward_step);
  last_revisit_signal_ = 0.0f;

  if (maze_.isWall(nx, ny)) {
    out.wall_hit = true;
    rat_.setWallHitLastAction(true);
    reward += static_cast<float>(cfg_.reward_wall_hit);
  } else {
    out.moved = true;
    rat_.setWallHitLastAction(false);
    rat_.setPosition(nx, ny);
    rat_.advanceWalkFrame();

    const size_t cell = static_cast<size_t>(ny) * maze_.width() + nx;
    const int64_t prev_visit = last_visit_step_[cell];
    last_visit_step_[cell] = static_cast<int64_t>(lifetime_steps_ + 1);

    if (prev_visit >= 0) {
      // Steps since the previous entry of this cell.
      const uint64_t k = (lifetime_steps_ + 1) - static_cast<uint64_t>(prev_visit);
      // Graded observation signal (0..1), see observation.h.
      last_revisit_signal_ = static_cast<float>(
          std::max(0.0, 1.0 - static_cast<double>(k) / 8.0));
      // Penalty when oscillating within the configured window.
      if (k <= static_cast<uint64_t>(cfg_.revisit_window_steps)) {
        reward += static_cast<float>(cfg_.reward_revisit);
      }
    }

    // Cheese collection.
    for (const Position& c : cheeses_) {
      if (c.x == nx && c.y == ny) {
        out.cheese_reached = true;
        reward += static_cast<float>(cfg_.reward_cheese);
        break;
      }
    }
  }

  // Homeostasis update driven by real events.
  Homeostasis::Events he;
  he.moved = out.moved;
  he.wall_hit = out.wall_hit;
  he.cheese = out.cheese_reached;
  homeo_.update(he);

  last_reward_ = reward;
  ++lifetime_steps_;
  if (!out.cheese_reached) ++steps_since_cheese_;

  pushFrame();

  if (out.cheese_reached) {
    ++cheese_total_;
    steps_since_cheese_ = 0;
    // New target; periodically regenerate the whole maze. Learning weights
    // are untouched (the agent persists independently).
    if (cheese_total_ % static_cast<uint64_t>(cfg_.maze_regenerate_every_cheeses) ==
        0) {
      regenerateMaze();
      out.maze_regenerated = true;
    } else {
      placeCheese();
      pushFrame();
    }
  }

  out.homeo_after = homeo_.snapshot();
  out.reward_nav = reward;
  return out;
}

}  // namespace sir
