#include "organism/homeostasis.h"

#include <algorithm>

namespace sir {

Homeostasis::Homeostasis(const HomeoParams& p) : p_(p) {}

void Homeostasis::update(const Events& e) {
  // Energy: base metabolism; movement costs extra; cheese restores.
  energy_ -= p_.energy_cost_step;
  if (e.moved) energy_ -= p_.energy_cost_move;
  if (e.cheese) energy_ += p_.energy_cheese_gain;

  // Hunger grows with time; cheese reduces it.
  hunger_ += p_.hunger_rate;
  if (e.cheese) hunger_ -= p_.hunger_cheese_reduction;

  // Fatigue: movement tires; resting (consolidation) or idling recovers.
  if (e.resting) {
    fatigue_ -= p_.fatigue_recovery_rest;
  } else if (e.moved) {
    fatigue_ += p_.fatigue_per_move;
  } else {
    fatigue_ -= p_.fatigue_recovery_idle;
  }

  // Stress: collisions raise it (worse when repeated); success and cheese
  // relieve it; passive decay toward 0.
  if (e.wall_hit) {
    ++wall_streak_;
    stress_ += p_.stress_wall_hit + (wall_streak_ >= 2 ? p_.stress_repeat_bonus : 0.0f);
    stress_free_steps_ = 0;
  } else {
    if (e.moved) {
      wall_streak_ = 0;
      stress_ -= p_.stress_success_relief;
    }
    if (e.cheese) stress_ -= p_.stress_cheese_relief;
    stress_ -= p_.stress_recovery_rate;
    if (stress_free_steps_ < 1000000) ++stress_free_steps_;
  }

  // Curiosity need grows slowly and is consumed by novelty experience.
  curiosity_need_ += p_.curiosity_need_rate;

  // Satisfaction: cheese is satisfying; otherwise reverts toward 0.5.
  if (e.cheese) {
    satisfaction_ += p_.satisfaction_cheese;
  } else {
    satisfaction_ += (0.5f - satisfaction_) * p_.satisfaction_reversion;
  }

  energy_ = clamp01(energy_);
  hunger_ = clamp01(hunger_);
  fatigue_ = clamp01(fatigue_);
  stress_ = clamp01(stress_);
  curiosity_need_ = clamp01(curiosity_need_);
  satisfaction_ = clamp01(satisfaction_);
  uncertainty_ = clamp01(uncertainty_);
}

void Homeostasis::applyNovelty(float novelty) {
  if (novelty > 0.0f) {
    curiosity_need_ = clamp01(curiosity_need_ - p_.curiosity_need_reduce * novelty);
  }
}

void Homeostasis::setUncertainty(float u) { uncertainty_ = clamp01(u); }

Homeostasis::Snapshot Homeostasis::snapshot() const {
  return {energy_, hunger_, fatigue_, stress_, curiosity_need_, satisfaction_,
          uncertainty_};
}

void Homeostasis::restore(const Snapshot& s) {
  energy_ = clamp01(s.energy);
  hunger_ = clamp01(s.hunger);
  fatigue_ = clamp01(s.fatigue);
  stress_ = clamp01(s.stress);
  curiosity_need_ = clamp01(s.curiosity_need);
  satisfaction_ = clamp01(s.satisfaction);
  uncertainty_ = clamp01(s.uncertainty);
}

}  // namespace sir
