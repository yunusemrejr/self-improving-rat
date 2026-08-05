#pragma once
// The simulation orchestrates maze, rat, cheese, homeostasis and the
// navigation reward. It is deterministic given the RNG seed and the sequence
// of actions chosen by the agent. It never uses pathfinding or privileged
// information. Homeostasis is owned here (the body lives in the maze), but
// the class is implemented in the organism module (src/organism/).

#include "organism/homeostasis.h"
#include "simulation/maze.h"
#include "simulation/observation.h"
#include "simulation/rat.h"
#include "utility/config.h"
#include "utility/rng.h"

#include <cstdint>
#include <vector>

namespace sir {

class Simulation {
 public:
  Simulation(const Config& cfg, Rng& rng);

  struct StepOutcome {
    float reward_nav = 0.0f;  // navigation component (cheese/wall/step/revisit)
    bool cheese_reached = false;
    bool maze_regenerated = false;
    bool wall_hit = false;
    bool moved = false;
    Homeostasis::Snapshot homeo_before;
    Homeostasis::Snapshot homeo_after;
  };

  // Applies one action; advances rat/cheese/maze/homeostasis lifecycle.
  // The composite reward (homeostasis + curiosity + prediction terms) is
  // composed by the application from the outcome; only the navigation part
  // is returned here.
  StepOutcome step(Action a);

  // Forces a new maze (keyboard shortcut R). Never touches learned weights.
  void regenerateMaze();

  // Current observation: the last `observation_frames` base frames flattened
  // (oldest first). Size = observationInputSize().
  const float* observe() const { return frame_history_.data(); }
  int observationInputSize() const { return input_size_; }

  // External signals that feed the observation vector (set by the app before
  // the next observe()/step()).
  void setNoveltySignal(float v) { novelty_signal_ = v; }
  void setResting(bool v) { resting_ = v; }

  // True when the last executed step re-entered a cell visited within the
  // revisit window (real observation signal, used by metrics).
  bool lastRevisit() const { return last_revisit_signal_ > 0.0f; }

  // Accessors for rendering and diagnostics.
  const Maze& maze() const { return maze_; }
  const Rat& rat() const { return rat_; }
  const std::vector<Position>& cheeses() const { return cheeses_; }
  Homeostasis& homeostasis() { return homeo_; }
  const Homeostasis& homeostasis() const { return homeo_; }

  // Lifetime counters (survive maze regenerations; restored from checkpoints
  // by the caller).
  uint64_t lifetimeSteps() const { return lifetime_steps_; }
  uint64_t cheeseTotal() const { return cheese_total_; }
  uint32_t mazeGenerations() const { return maze_generations_; }
  int stepsSinceLastCheese() const { return steps_since_cheese_; }

  void setLifetimeSteps(uint64_t v) { lifetime_steps_ = v; }
  void setCheeseTotal(uint64_t v) { cheese_total_ = v; }
  void setMazeGenerations(uint32_t v) { maze_generations_ = v; }

 private:
  void placeCheese();
  void resetRatAndCheese();  // fresh positions on the current maze
  void pushFrame();

  const Config& cfg_;
  Rng& rng_;

  Maze maze_;
  Rat rat_;
  Homeostasis homeo_;
  std::vector<Position> cheeses_;

  std::vector<int64_t> last_visit_step_;  // per cell: last step the cell was entered
  std::vector<float> frame_history_;  // stacked observations (oldest first)
  int input_size_ = 0;

  uint64_t lifetime_steps_ = 0;
  uint64_t cheese_total_ = 0;
  uint32_t maze_generations_ = 0;
  int steps_since_cheese_ = 0;
  float last_reward_ = 0.0f;
  float last_revisit_signal_ = 0.0f;
  float novelty_signal_ = 0.0f;
  bool resting_ = false;
  bool first_frame_ = true;
};

}  // namespace sir
