// Utility-layer tests: config parsing/clamping, RNG determinism and state
// round-trip, RingBuffer bounds, Metrics bounds.

#include "test_framework.h"

#include "utility/config.h"
#include "utility/logger.h"
#include "utility/ring_buffer.h"
#include "utility/rng.h"
#include "utility/rolling_stats.h"

#include <cstdio>
#include <string>

using namespace sir;

TEST(config_parses_and_clamps) {
  Config cfg;
  std::string err, warns;
  // Write a temporary config file.
  const char* path = "/tmp/sir_test_config.cfg";
  FILE* f = std::fopen(path, "w");
  CHECK(f != nullptr);
  std::fputs("window_width = 99999\n", f);        // clamped to 4096
  std::fputs("maze_width = 3\n", f);              // clamped to 5
  std::fputs("learning_rate = 5.0\n", f);         // clamped to 0.1
  std::fputs("maze_braid_probability = 2.5\n", f); // clamped to 1.0
  std::fputs("debug_display = true\n", f);
  std::fputs("unknown_key = 1\n", f);             // warning
  std::fputs("batch_size = notanumber\n", f);     // warning
  std::fputs("checkpoint_dir = /tmp/sir_ckpts\n", f);
  std::fclose(f);

  const int applied = cfg.loadFromFile(path, &err, &warns);
  CHECK(applied > 0);
  CHECK(cfg.window_width == 4096);
  CHECK(cfg.maze_width == 5);
  CHECK_NEAR(cfg.learning_rate, 0.1, 1e-9);
  CHECK_NEAR(cfg.maze_braid_probability, 1.0, 1e-9);
  CHECK(cfg.debug_display == true);
  CHECK(cfg.checkpoint_dir == "/tmp/sir_ckpts");
  CHECK(!warns.empty());  // unknown key + invalid batch size warned
}

TEST(config_missing_file) {
  Config cfg;
  std::string err;
  CHECK(cfg.loadFromFile("/nonexistent/sir_config.cfg", &err, nullptr) == -1);
  CHECK(!err.empty());
  // Defaults remain intact.
  CHECK_NEAR(cfg.learning_rate, 0.0003, 1e-12);
}

TEST(rng_deterministic_same_seed) {
  Rng a(123), b(123);
  for (int i = 0; i < 1000; ++i) {
    CHECK(a.nextU32() == b.nextU32());
    CHECK_NEAR(a.uniform01(), b.uniform01(), 1e-15);
  }
}

TEST(rng_state_roundtrip) {
  Rng r(42);
  for (int i = 0; i < 500; ++i) r.nextU32();
  const std::string state = r.saveState();
  Rng r2(999);  // different seed
  CHECK(r2.restoreState(state));
  for (int i = 0; i < 200; ++i) CHECK(r.nextU32() == r2.nextU32());
  // A corrupt state is rejected.
  Rng r3(1);
  CHECK(!r3.restoreState("garbage !!! not an mt19937 state"));
}

TEST(rng_uniform_int_bounds) {
  Rng r(7);
  for (int i = 0; i < 10000; ++i) {
    const int v = r.uniformInt(2, 5);
    CHECK(v >= 2 && v <= 5);
  }
}

TEST(ring_buffer_bounded_and_wraps) {
  RingBuffer<int> buf(4);
  CHECK(buf.empty());
  for (int i = 0; i < 4; ++i) buf.push(i);
  CHECK(buf.full());
  CHECK(buf.size() == 4);
  // Oldest first.
  CHECK(buf[0] == 0);
  CHECK(buf[3] == 3);
  buf.push(4);  // wraps
  CHECK(buf.size() == 4);
  CHECK(buf[0] == 1);  // 0 evicted
  CHECK(buf[3] == 4);
  buf.clear();
  CHECK(buf.empty());
}

TEST(metrics_bounded_and_event_derived) {
  Metrics m;
  Homeostasis::Snapshot h;
  for (int i = 0; i < 5000; ++i) {
    m.recordStep(-0.02f, i % 3 == 0, i % 7 == 0, i % 4, (i - 1) % 4,
                 i % 11 == 0, h, 0.1f, 0.01f, 0.2f);
  }
  CHECK_NEAR(m.recentAvgReward(), -0.02, 1e-9);
  CHECK(m.wallHitsTotal() > 0);
  CHECK(m.wallRatePer1000Steps() > 0.0);
  CHECK(m.actionEntropy() >= 0.0 && m.actionEntropy() <= 1.0);
  CHECK(m.repeatedActionRate() >= 0.0 && m.repeatedActionRate() <= 1.0);
  CHECK(m.explorationRate() >= 0.0 && m.explorationRate() <= 1.0);
  CHECK(m.avgEnergy() >= 0.0 && m.avgEnergy() <= 1.0);
  CHECK(m.minEnergy() >= 0.0 && m.minEnergy() <= 1.0);
  // Median of 1..3 = 2.
  m.recordCheese(1);
  m.recordCheese(2);
  m.recordCheese(3);
  CHECK_NEAR(m.medianStepsPerCheese(), 2.0, 1e-9);
}

TEST(logger_rotates_at_limit) {
  const std::string path = "/tmp/sir_test_log.txt";
  std::remove(path.c_str());
  std::remove((path + ".1").c_str());
  Logger log;
  CHECK(log.open(path, 2048));
  for (int i = 0; i < 300; ++i) log.info("padding line to force rotation 0123456789");
  CHECK(std::fopen((path + ".1").c_str(), "r") != nullptr);
  log.info("still appending after rotation");
}
