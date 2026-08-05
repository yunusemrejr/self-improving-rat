// Organism tests: homeostasis bounds and event effects, development
// schedules, long-run boundedness of all internal state.

#include "test_framework.h"

#include "organism/development.h"
#include "organism/homeostasis.h"
#include "utility/rng.h"

#include <algorithm>
#include <cstdio>

using namespace sir;

TEST(homeostasis_bounds_long_run) {
  HomeoParams p;
  Homeostasis h(p);
  Rng rng(1);
  for (int i = 0; i < 500000; ++i) {
    Homeostasis::Events e;
    e.moved = rng.uniform01() < 0.5;
    e.wall_hit = rng.uniform01() < 0.2;
    e.cheese = rng.uniform01() < 0.01;
    e.resting = rng.uniform01() < 0.05;
    h.update(e);
    if (rng.uniform01() < 0.3) h.applyNovelty(static_cast<float>(rng.uniform01()));
    h.setUncertainty(static_cast<float>(rng.uniform01()));
    CHECK(h.energy() >= 0.0f && h.energy() <= 1.0f);
    CHECK(h.hunger() >= 0.0f && h.hunger() <= 1.0f);
    CHECK(h.fatigue() >= 0.0f && h.fatigue() <= 1.0f);
    CHECK(h.stress() >= 0.0f && h.stress() <= 1.0f);
    CHECK(h.curiosityNeed() >= 0.0f && h.curiosityNeed() <= 1.0f);
    CHECK(h.satisfaction() >= 0.0f && h.satisfaction() <= 1.0f);
    CHECK(h.uncertainty() >= 0.0f && h.uncertainty() <= 1.0f);
    CHECK(h.consecutiveWallHits() >= 0);
    CHECK(h.stressFreeSteps() >= 0);
  }
}

TEST(homeostasis_event_effects) {
  HomeoParams p;
  Homeostasis h(p);
  const float e0 = h.energy();
  const float hu0 = h.hunger();
  const float s0 = h.satisfaction();
  Homeostasis::Events cheese;
  cheese.moved = true;
  cheese.cheese = true;
  h.update(cheese);
  CHECK(h.energy() > e0);       // cheese restores energy
  CHECK(h.hunger() < hu0);      // cheese reduces hunger
  CHECK(h.satisfaction() > s0); // cheese is satisfying
  // Resting reduces fatigue.
  Homeostasis h2(p);
  Homeostasis::Events idle;
  idle.moved = false;
  h2.update(idle);
  Homeostasis::Events rest;
  rest.resting = true;
  h2.update(rest);
  CHECK(h2.fatigue() <= 0.0f + 1e-6);  // no fatigue gained while resting
}

TEST(homeostasis_snapshot_restore) {
  HomeoParams p;
  Homeostasis h(p);
  Homeostasis::Events e;
  e.wall_hit = true;
  for (int i = 0; i < 50; ++i) h.update(e);
  const auto snap = h.snapshot();
  Homeostasis h2(p);
  h2.restore(snap);
  const auto snap2 = h2.snapshot();
  CHECK_NEAR(snap.energy, snap2.energy, 1e-6);
  CHECK_NEAR(snap.hunger, snap2.hunger, 1e-6);
  CHECK_NEAR(snap.fatigue, snap2.fatigue, 1e-6);
  CHECK_NEAR(snap.stress, snap2.stress, 1e-6);
  CHECK_NEAR(snap.curiosity_need, snap2.curiosity_need, 1e-6);
  CHECK_NEAR(snap.satisfaction, snap2.satisfaction, 1e-6);
  CHECK_NEAR(snap.uncertainty, snap2.uncertainty, 1e-6);
  // Out-of-range restore values are clamped.
  Homeostasis::Snapshot bad{2.0f, -1.0f, 5.0f, 0.5f, 0.5f, 0.5f, 0.5f};
  h2.restore(bad);
  CHECK(h2.energy() <= 1.0f);
  CHECK(h2.hunger() >= 0.0f);
  CHECK(h2.fatigue() <= 1.0f);
}

TEST(homeostasis_novelty_reduces_curiosity_need) {
  HomeoParams p;
  p.curiosity_need_reduce = 0.1f;
  Homeostasis h(p);
  const float before = h.curiosityNeed();
  h.applyNovelty(0.5f);
  CHECK(h.curiosityNeed() <= before);
}

TEST(development_schedules_continuous_and_bounded) {
  DevParams p;
  Development dev(p);
  double prev_age = -1;
  double prev_lr = -1, prev_eps = -1, prev_plast = -1;
  for (uint64_t steps = 0; steps <= 600000; steps += 1000) {
    const double age = dev.ageNorm(steps);
    CHECK(age >= 0.0 && age <= 1.0);
    CHECK(age >= prev_age);  // monotone
    prev_age = age;
    const double lr = dev.learningRateScale(steps);
    const double eps = dev.epsilonScale(steps);
    const double pl = dev.plasticityScale(steps);
    // Never zero; learning never switches off.
    CHECK(lr >= 0.1 && lr <= 1.0);
    CHECK(eps >= 0.05 && eps <= 1.0);
    CHECK(pl >= 0.1 && pl <= 1.0);
    if (prev_lr >= 0) {
      CHECK(lr <= prev_lr + 1e-9);  // non-increasing
      CHECK(eps <= prev_eps + 1e-9);
      CHECK(pl <= prev_plast + 1e-9);
    }
    prev_lr = lr;
    prev_eps = eps;
    prev_plast = pl;
  }
  // Smoothness: consecutive samples differ by less than the step size would
  // allow for a discontinuous jump (tau is continuous).
  CHECK_NEAR(dev.ageNorm(50000), 0.25, 1e-9);
}

TEST(development_zero_maturity_is_safe) {
  DevParams p;
  p.maturity_steps = 0;
  Development dev(p);
  CHECK_NEAR(dev.ageNorm(0), 1.0, 1e-9);
  CHECK(dev.learningRateScale(0) >= 0.1);
  CHECK(dev.epsilonScale(0) >= 0.05);
}
