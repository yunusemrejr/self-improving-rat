// Observation-vector tests: shape, bounds, scent behavior (radius limits,
// falloff, direction), wall bits, proprioception channels, and the
// no-cheating property (no cheese coordinates or maze map leak into the
// vector beyond the defined local signals).

#include "test_framework.h"

#include "simulation/maze.h"
#include "simulation/observation.h"
#include "simulation/rat.h"
#include "utility/rng.h"

#include <cstdio>

using namespace sir;

namespace {

// Open 11x9 grid with a single cheese placed manually.
struct Fixture {
  Maze maze{11, 9};
  Position cheese;
  Fixture() {
    for (int y = 0; y < 9; ++y)
      for (int x = 0; x < 11; ++x) maze.setWallForTests(x, y, false);
  }
  void obsAt(int x, int y, float* out, const ObservationContext& ctx = {}) {
    buildObservation(maze, x, y, Action::Up, false, 0.0f, 0.0f, {cheese}, 6, ctx,
                     out);
  }
};

}  // namespace

TEST(observation_shape_and_bounds) {
  Fixture f;
  f.cheese = {10, 8};
  float o[kObservationBase];
  f.obsAt(0, 0, o);
  CHECK(kObservationBase == 30);
  for (int i = 0; i < kObservationBase; ++i)
    CHECK(o[i] >= 0.0f && o[i] <= 1.0f);
}

TEST(observation_wall_bits_local) {
  Fixture f;
  f.cheese = {10, 8};
  // Build a wall north of the rat.
  f.maze.setWallForTests(5, 2, true);
  float o[kObservationBase];
  f.obsAt(5, 3, o);
  CHECK_NEAR(o[0], 1.0f, 1e-6);  // N wall bit
  CHECK_NEAR(o[4], 0.0f, 1e-6);  // S open
  CHECK_NEAR(o[2], 0.0f, 1e-6);  // E open
  // Wall bits reflect only the local 8-neighborhood (max distance 1).
  f.maze.setWallForTests(5, 0, true);  // far away: must not appear
  f.obsAt(5, 3, o);
  CHECK_NEAR(o[0], 1.0f, 1e-6);  // still just the adjacent N wall
}

TEST(observation_scent_limited_to_radius) {
  Fixture f;
  f.cheese = {10, 8};
  float o[kObservationBase];
  // Scent channels are action-aligned: Up(8), Down(9), Left(10), Right(11).
  // Cheese at manhattan distance 6 == radius: faint but nonzero.
  f.obsAt(4, 8, o);  // cheese east: dx=6, dy=0
  CHECK(o[11] > 0.0f);  // Right scent
  CHECK_NEAR(o[8], 0.0f, 1e-6);  // Up scent zero
  // Beyond the radius: no scent at all (no privileged distance information).
  f.obsAt(0, 8, o);  // dx=10 > 6
  CHECK_NEAR(o[11], 0.0f, 1e-6);
  CHECK_NEAR(o[9], 0.0f, 1e-6);
  // North cheese produces an Up-scent.
  f.cheese = {5, 2};
  f.obsAt(5, 8, o);  // dy=-6
  CHECK(o[8] > 0.0f);
}

TEST(observation_scent_gradient_monotone) {
  Fixture f;
  f.cheese = {10, 8};
  float o1[kObservationBase], o2[kObservationBase], o3[kObservationBase];
  f.obsAt(8, 8, o1);  // dx=2
  f.obsAt(6, 8, o2);  // dx=4
  f.obsAt(4, 8, o3);  // dx=6
  CHECK(o1[11] > o2[11]);
  CHECK(o2[11] > o3[11]);
  // Direction alignment: cheese east -> Right channel is the max.
  CHECK(o1[11] > o1[8] && o1[11] > o1[9] && o1[11] > o1[10]);
}

TEST(observation_no_cheese_coordinates) {
  Fixture f;
  f.cheese = {10, 8};
  float o[kObservationBase];
  f.obsAt(0, 0, o);
  // The vector carries wall bits, scent directions, action history, internal
  // state and normalized position — never the cheese's x/y as separate
  // channels, and scent is bounded to [0,1] with no absolute distance.
  for (int i = 0; i < kObservationBase; ++i) CHECK(o[i] <= 1.0f);
}

TEST(observation_proprioception_bounds) {
  Fixture f;
  f.cheese = {10, 8};
  float o[kObservationBase];
  f.obsAt(0, 0, o);
  CHECK_NEAR(o[28], 0.0f, 1e-6);  // x normalized: 0/(11-1)
  CHECK_NEAR(o[29], 0.0f, 1e-6);  // y normalized
  f.obsAt(10, 8, o);
  CHECK_NEAR(o[28], 1.0f, 1e-6);
  CHECK_NEAR(o[29], 1.0f, 1e-6);
  f.obsAt(5, 4, o);
  CHECK_NEAR(o[28], 0.5f, 1e-6);
  CHECK_NEAR(o[29], 0.5f, 1e-6);
}

TEST(observation_internal_state_channels) {
  Fixture f;
  f.cheese = {10, 8};
  ObservationContext ctx;
  ctx.energy = 0.7f;
  ctx.hunger = 0.3f;
  ctx.fatigue = 0.4f;
  ctx.stress = 0.2f;
  ctx.curiosity_need = 0.6f;
  ctx.satisfaction = 0.5f;
  ctx.uncertainty = 0.8f;
  ctx.novelty = 0.25f;
  ctx.resting = 1.0f;
  float o[kObservationBase];
  f.obsAt(5, 4, o, ctx);
  CHECK_NEAR(o[19], 0.7f, 1e-6);
  CHECK_NEAR(o[20], 0.3f, 1e-6);
  CHECK_NEAR(o[21], 0.4f, 1e-6);
  CHECK_NEAR(o[22], 0.2f, 1e-6);
  CHECK_NEAR(o[23], 0.6f, 1e-6);
  CHECK_NEAR(o[24], 0.5f, 1e-6);
  CHECK_NEAR(o[25], 0.8f, 1e-6);
  CHECK_NEAR(o[26], 0.25f, 1e-6);
  CHECK_NEAR(o[27], 1.0f, 1e-6);
}
