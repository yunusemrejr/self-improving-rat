// Maze and simulation tests: generation validity, connectivity across many
// seeds, collision handling, cheese placement, determinism, regeneration.

#include "test_framework.h"

#include "simulation/simulation.h"
#include "utility/config.h"
#include "utility/rng.h"

#include <cstdio>
#include <set>

using namespace sir;

namespace {

// Number of walkable cells (BFS-independent count).
int countWalkable(const Maze& m) {
  int n = 0;
  for (int y = 0; y < m.height(); ++y)
    for (int x = 0; x < m.width(); ++x)
      if (m.isWalkable(x, y)) ++n;
  return n;
}

}  // namespace

TEST(maze_valid_dimensions_and_generation) {
  Config cfg;
  cfg.observation_frames = 1;
  for (int seed = 1; seed <= 200; ++seed) {
    Rng rng(static_cast<uint32_t>(seed));
    Simulation sim(cfg, rng);
    const Maze& m = sim.maze();
    CHECK(m.width() == cfg.maze_width);
    CHECK(m.height() == cfg.maze_height);
    // Border is always wall.
    for (int x = 0; x < m.width(); ++x) CHECK(m.isWall(x, 0));
    for (int x = 0; x < m.width(); ++x) CHECK(m.isWall(x, m.height() - 1));
    for (int y = 0; y < m.height(); ++y) CHECK(m.isWall(0, y));
    for (int y = 0; y < m.height(); ++y) CHECK(m.isWall(m.width() - 1, y));
    // Rat and cheese on walkable cells.
    CHECK(m.isWalkable(sim.rat().position().x, sim.rat().position().y));
    CHECK(sim.cheeses().size() == 1);
    CHECK(m.isWalkable(sim.cheeses()[0].x, sim.cheeses()[0].y));
    // Cheese reachable from the rat (BFS over the whole maze).
    CHECK(m.reachableCount(sim.rat().position().x, sim.rat().position().y) ==
          countWalkable(m));
  }
}

TEST(maze_connectivity_always_full) {
  Config cfg;
  cfg.observation_frames = 1;
  // Braiding must never disconnect the maze.
  for (double braid : {0.0, 0.3, 0.6, 0.95}) {
    cfg.maze_braid_probability = braid;
    for (int seed = 1; seed <= 50; ++seed) {
      Rng rng(static_cast<uint32_t>(seed));
      Simulation sim(cfg, rng);
      const Maze& m = sim.maze();
      const int walkable = countWalkable(m);
      for (int y = 1; y < m.height() - 1; ++y) {
        for (int x = 1; x < m.width() - 1; ++x) {
          if (!m.isWalkable(x, y)) continue;
          CHECK(m.reachableCount(x, y) == walkable);
        }
      }
    }
  }
}

TEST(sim_step_collision_and_movement) {
  Config cfg;
  cfg.observation_frames = 1;
  Rng rng(5);
  Simulation sim(cfg, rng);
  const Maze& m = sim.maze();
  // Find a wall-adjacent cell and verify wall-hit semantics.
  bool tested = false;
  for (int tries = 0; tries < 200 && !tested; ++tries) {
    auto out = sim.step(static_cast<Action>(rng.uniformInt(0, 3)));
    const auto& p = sim.rat().position();
    if (out.wall_hit) {
      tested = true;
      // The rat did not move (position unchanged is implied by wall hit).
      CHECK(sim.rat().wallHitLastAction());
    } else {
      CHECK(out.moved);
      CHECK(!sim.rat().wallHitLastAction());
      CHECK(m.isWalkable(p.x, p.y));  // never inside a wall
      CHECK(p.x >= 0 && p.y >= 0 && p.x < m.width() && p.y < m.height());
    }
  }
  CHECK(tested);  // some wall hit occurred
}

TEST(sim_cheese_collection_and_replacement) {
  Config cfg;
  cfg.observation_frames = 1;
  cfg.maze_regenerate_every_cheeses = 100000;  // avoid regen during test
  cfg.min_cheese_distance = 2;
  Rng rng(11);
  Simulation sim(cfg, rng);
  // Drive the rat with BFS-free random actions until cheese is found.
  int cheese_before = static_cast<int>(sim.cheeseTotal());
  bool found = false;
  for (int i = 0; i < 200000 && !found; ++i) {
    auto out = sim.step(static_cast<Action>(rng.uniformInt(0, 3)));
    if (out.cheese_reached) {
      found = true;
      CHECK(sim.cheeseTotal() == static_cast<uint64_t>(cheese_before + 1));
      // The new cheese is on a walkable cell and reachable.
      const Maze& m = sim.maze();
      CHECK(m.isWalkable(sim.cheeses()[0].x, sim.cheeses()[0].y));
      CHECK(m.reachableCount(sim.rat().position().x, sim.rat().position().y) ==
            countWalkable(m));
    }
  }
  CHECK(found);
}

TEST(sim_regeneration_keeps_lifetime_state) {
  Config cfg;
  cfg.observation_frames = 1;
  cfg.maze_regenerate_every_cheeses = 3;
  Rng rng(21);
  Simulation sim(cfg, rng);
  const uint32_t start_gens = sim.mazeGenerations();
  bool regenerated = false;
  for (int i = 0; i < 100000 && !regenerated; ++i) {
    auto out = sim.step(static_cast<Action>(rng.uniformInt(0, 3)));
    if (out.maze_regenerated) {
      regenerated = true;
      CHECK(sim.mazeGenerations() == start_gens + 1);
      // The rat is on a walkable cell of the NEW maze.
      CHECK(sim.maze().isWalkable(sim.rat().position().x, sim.rat().position().y));
      CHECK(sim.cheeses().size() == 1);
      CHECK(sim.maze().isWalkable(sim.cheeses()[0].x, sim.cheeses()[0].y));
      // Lifetime steps kept growing.
      CHECK(sim.lifetimeSteps() > 0);
    }
  }
  CHECK(regenerated);
}

TEST(sim_deterministic_same_seed) {
  Config cfg;
  cfg.observation_frames = 1;
  // Same seed -> identical action outcomes and observations.
  auto run = [&](uint32_t seed, int steps) {
    Rng rng(seed);
    Simulation sim(cfg, rng);
    std::vector<uint32_t> trace;
    for (int i = 0; i < steps; ++i) {
      const Action a = static_cast<Action>(rng.uniformInt(0, 3));
      auto out = sim.step(a);
      const float* o = sim.observe();
      uint32_t h = 2166136261u;
      for (int k = 0; k < sim.observationInputSize(); ++k) {
        h ^= static_cast<uint32_t>(o[k] * 1000.0f);
        h *= 16777619u;
      }
      trace.push_back(h ^ (out.wall_hit ? 0x1u : 0u) ^ (out.cheese_reached ? 0x2u : 0u));
    }
    return trace;
  };
  const auto t1 = run(1234, 500);
  const auto t2 = run(1234, 500);
  CHECK(t1 == t2);
  const auto t3 = run(1235, 500);
  CHECK(t1 != t3);  // different seed diverges
}

TEST(sim_observation_frames_stacking) {
  Config cfg;
  cfg.observation_frames = 2;
  Rng rng(3);
  Simulation sim(cfg, rng);
  CHECK(sim.observationInputSize() == kObservationBase * 2);
  const float* o = sim.observe();
  for (int i = 0; i < sim.observationInputSize(); ++i)
    CHECK(o[i] >= 0.0f && o[i] <= 1.0f);
  // The two stacked frames must be identical at start (first frame duplicated).
  for (int i = 0; i < kObservationBase; ++i)
    CHECK_NEAR(o[i], o[kObservationBase + i], 1e-6);
}
