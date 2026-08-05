#pragma once
// Bounded internal homeostatic variables (energy, hunger, fatigue, stress,
// curiosity need, satisfaction, prediction uncertainty). All values live in
// [0,1], updated by deterministic rules driven by real simulation events.
// They feed the observation vector and the composite reward; behavior must
// emerge from learned responses, never from hardcoded "moods".

#include <cstdint>

namespace sir {

// Homeostasis parameters (mirrors the [organism] config group).
struct HomeoParams {
  float energy_cost_step = 0.0004f;      // base metabolism per step
  float energy_cost_move = 0.0008f;      // extra cost per successful move
  float energy_cheese_gain = 0.35f;      // cheese restores energy
  float hunger_rate = 0.0003f;           // hunger grows over time
  float hunger_cheese_reduction = 0.5f;  // cheese reduces hunger
  float fatigue_per_move = 0.0006f;
  float fatigue_recovery_rest = 0.004f;  // per consolidation step
  float fatigue_recovery_idle = 0.0002f; // per step without movement
  float stress_wall_hit = 0.04f;
  float stress_repeat_bonus = 0.03f;     // extra stress for repeated wall hits
  float stress_success_relief = 0.002f;  // successful navigation calms
  float stress_cheese_relief = 0.2f;
  float stress_recovery_rate = 0.0005f;  // passive decay toward 0
  float curiosity_need_rate = 0.0002f;   // novelty need grows with time
  float curiosity_need_reduce = 0.02f;   // per unit of novelty experienced
  float satisfaction_cheese = 0.4f;
  float satisfaction_reversion = 0.001f; // mean reversion toward 0.5
};

class Homeostasis {
 public:
  explicit Homeostasis(const HomeoParams& p);

  // One simulation step. `events` describe what actually happened.
  struct Events {
    bool moved = false;
    bool wall_hit = false;
    bool cheese = false;
    bool resting = false;  // consolidation rest tick (no movement)
  };
  void update(const Events& e);

  // External signals from the learning system.
  void applyNovelty(float novelty);          // reduces curiosity need
  void setUncertainty(float u);              // prediction uncertainty in [0,1]

  struct Snapshot {
    float energy = 0.8f;
    float hunger = 0.2f;
    float fatigue = 0.0f;
    float stress = 0.0f;
    float curiosity_need = 0.5f;
    float satisfaction = 0.5f;
    float uncertainty = 0.5f;
  };
  Snapshot snapshot() const;
  void restore(const Snapshot& s);

  // Accessors (also used by the observation builder).
  float energy() const { return energy_; }
  float hunger() const { return hunger_; }
  float fatigue() const { return fatigue_; }
  float stress() const { return stress_; }
  float curiosityNeed() const { return curiosity_need_; }
  float satisfaction() const { return satisfaction_; }
  float uncertainty() const { return uncertainty_; }

  int consecutiveWallHits() const { return wall_streak_; }
  // Steps since the last stress increase (stress recovery time; transient).
  int stressFreeSteps() const { return stress_free_steps_; }

 private:
  float clamp01(float v) const { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

  HomeoParams p_;
  float energy_ = 0.8f;
  float hunger_ = 0.2f;
  float fatigue_ = 0.0f;
  float stress_ = 0.0f;
  float curiosity_need_ = 0.5f;
  float satisfaction_ = 0.5f;
  float uncertainty_ = 0.5f;
  int wall_streak_ = 0;
  int stress_free_steps_ = 0;
};

}  // namespace sir
