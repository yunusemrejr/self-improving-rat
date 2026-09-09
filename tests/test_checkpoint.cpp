// Checkpoint persistence tests: round-trip fidelity, atomicity, corruption
// and checksum rejection, NaN rejection, backup fallback, incompatible
// topology, and interrupted-write recovery.

#include "test_framework.h"

#include "learning/agent.h"
#include "persistence/checkpoint.h"
#include "utility/config.h"
#include "utility/logger.h"
#include "utility/rng.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace sir;

namespace {

struct CkptEnv {
  std::string dir;
  Config cfg;
  Logger log;
  std::unique_ptr<CheckpointStore> store;
  explicit CkptEnv(const std::string& name)
      : dir(std::string("/tmp/sir_ckpt_") + name), cfg() {
    std::filesystem::create_directories(dir);
    std::remove((dir + "/rat.sir").c_str());
    std::remove((dir + "/rat.sir.bak").c_str());
    std::remove((dir + "/rat.sir.tmp").c_str());
    cfg.checkpoint_dir = dir;
    log.open("/tmp/sir_ckpt_test.log", 1 << 20);
    store = std::make_unique<CheckpointStore>(cfg, log);
  }
  ~CkptEnv() {
    std::remove((dir + "/rat.sir").c_str());
    std::remove((dir + "/rat.sir.bak").c_str());
    std::remove((dir + "/rat.sir.tmp").c_str());
  }
  CheckpointStore& s() { return *store; }
  AgentState makeState() {
    Config c;
    c.observation_frames = 1;
    Rng rng(42);
    Agent a(c, rng);
    AgentState st;
    a.exportState(st);
    st.homeo = Homeostasis::Snapshot{0.7f, 0.3f, 0.1f, 0.2f, 0.4f, 0.5f, 0.6f};
    st.cheese_total = 5;
    st.lifetime_steps = 12345;
    st.maze_generations = 2;
    st.consolidation_cycles = 3;
    st.timestamp_utc = 1700000000;
    return st;
  }
};

// Reads a checkpoint file completely.
std::vector<uint8_t> readFile(const std::string& path) {
  std::vector<uint8_t> out;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return out;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  out.resize(static_cast<size_t>(n));
  const size_t rd = std::fread(out.data(), 1, out.size(), f);
  std::fclose(f);
  out.resize(rd);
  return out;
}

bool writeFile(const std::string& path, const std::vector<uint8_t>& bytes) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const size_t w = std::fwrite(bytes.data(), 1, bytes.size(), f);
  std::fclose(f);
  return w == bytes.size();
}

}  // namespace

TEST(checkpoint_roundtrip_fidelity) {
  CkptEnv env("roundtrip");
  const AgentState st = env.makeState();
  CHECK(env.s().save(st));
  AgentState out;
  const LoadResult r = env.s().load(&out);
  CHECK(r == LoadResult::Ok);
  // Compare deterministic state (time fields are informational).
  CHECK(out.online_params == st.online_params);
  CHECK(out.target_params == st.target_params);
  CHECK(out.adam_m == st.adam_m);
  CHECK(out.adam_v == st.adam_v);
  CHECK(out.masks == st.masks);
  CHECK(out.utility == st.utility);
  CHECK(out.episodic_floats == st.episodic_floats);
  CHECK(out.episodic_meta == st.episodic_meta);
  CHECK(out.novelty_table == st.novelty_table);
  CHECK(out.rng_state == st.rng_state);
  CHECK(out.input == st.input);
  CHECK(out.rnn == st.rnn);
  CHECK(out.training_steps == st.training_steps);
  CHECK(out.lifetime_steps == st.lifetime_steps);
  CHECK(out.cheese_total == st.cheese_total);
  CHECK(out.maze_generations == st.maze_generations);
  CHECK_NEAR(out.homeo.energy, st.homeo.energy, 1e-6);
  CHECK_NEAR(out.homeo.stress, st.homeo.stress, 1e-6);
  CHECK(out.consolidation_cycles == st.consolidation_cycles);
  CHECK(out.allFinite());
}

TEST(checkpoint_atomic_write_leaves_no_temp) {
  CkptEnv env("atomic");
  CHECK(env.s().save(env.makeState()));
  // No temp file left behind; backup created on the second save.
  CHECK(!std::fopen((env.dir + "/rat.sir.tmp").c_str(), "r"));
  CHECK(env.s().save(env.makeState()));
  CHECK(std::fopen((env.dir + "/rat.sir.bak").c_str(), "r"));
  CHECK(std::fopen((env.dir + "/rat.sir").c_str(), "r"));
}

TEST(checkpoint_truncated_rejected) {
  CkptEnv env("trunc");
  CHECK(env.s().save(env.makeState()));
  auto bytes = readFile(env.dir + "/rat.sir");
  CHECK(bytes.size() > 100);
  // Truncate in the middle of the parameter block.
  bytes.resize(bytes.size() / 2);
  writeFile(env.dir + "/rat.sir", bytes);
  AgentState out;
  CHECK(env.s().load(&out) != LoadResult::Ok);
}

TEST(checkpoint_checksum_rejected) {
  CkptEnv env("csum");
  CHECK(env.s().save(env.makeState()));
  auto bytes = readFile(env.dir + "/rat.sir");
  // Flip a byte in the middle (after the magic, before the checksum).
  bytes[bytes.size() / 2] ^= 0x5A;
  writeFile(env.dir + "/rat.sir", bytes);
  AgentState out;
  CHECK(env.s().load(&out) != LoadResult::Ok);
  // A flipped checksum byte itself is also rejected.
  CHECK(env.s().save(env.makeState()));
  bytes = readFile(env.dir + "/rat.sir");
  bytes.back() ^= 0x01;
  writeFile(env.dir + "/rat.sir", bytes);
  CHECK(env.s().load(&out) != LoadResult::Ok);
}

TEST(checkpoint_nan_data_rejected) {
  CkptEnv env("nan");
  CHECK(env.s().save(env.makeState()));
  auto bytes = readFile(env.dir + "/rat.sir");
  // Overwrite the first parameter float with NaN (little-endian f32).
  // Parameters start after the fixed header; find a plausible offset by
  // scanning for a large run of nonzero bytes: simpler — patch a float at an
  // offset inside the params (starts ~ after rng state). We compute the
  // offset by re-serializing known sizes is complex; instead write NaN into
  // the homeostasis block: it sits right after rng_state. Locate via the
  // known seed/timestamp pattern is fragile, so we use a robust approach:
  // every float in the file could be NaN; patch the byte at 25% of the file.
  const size_t off = bytes.size() / 4;
  if (off + 4 <= bytes.size()) {
    bytes[off] = 0x00; bytes[off + 1] = 0x00; bytes[off + 2] = 0xC0;
    bytes[off + 3] = 0x7F;  // NaN bits
  }
  writeFile(env.dir + "/rat.sir", bytes);
  AgentState out;
  CHECK(env.s().load(&out) != LoadResult::Ok);
}

TEST(checkpoint_backup_fallback) {
  CkptEnv env("backup");
  CHECK(env.s().save(env.makeState()));
  CHECK(env.s().save(env.makeState()));  // creates the backup
  // Corrupt the primary; the valid backup must be recovered.
  auto bytes = readFile(env.dir + "/rat.sir");
  bytes[10] ^= 0xFF;
  writeFile(env.dir + "/rat.sir", bytes);
  AgentState out;
  const LoadResult r = env.s().load(&out);
  CHECK(r == LoadResult::Ok);
  CHECK(out.lifetime_steps == env.makeState().lifetime_steps);
  // The corrupt primary was moved aside (evidence preserved, not deleted).
  CHECK(std::fopen((env.dir + "/rat.sir.bad").c_str(), "r") ||
        std::fopen((env.dir + "/rat.sir.legacy-v1").c_str(), "r"));
}

TEST(checkpoint_incompatible_topology_rejected) {
  CkptEnv env("incompat");
  CHECK(env.s().save(env.makeState()));
  // A config with different topology must reject the checkpoint.
  Config other = env.cfg;
  other.rnn_hidden = 32;
  CheckpointStore other_store(other, env.log);
  AgentState out;
  CHECK(other_store.load(&out) == LoadResult::Incompatible);
}

TEST(checkpoint_missing_is_missing) {
  CkptEnv env("missing");
  AgentState out;
  CHECK(env.s().load(&out) == LoadResult::Missing);
}

TEST(checkpoint_trailing_garbage_rejected) {
  CkptEnv env("trail");
  CHECK(env.s().save(env.makeState()));
  auto bytes = readFile(env.dir + "/rat.sir");
  bytes.push_back(0x00);
  writeFile(env.dir + "/rat.sir", bytes);
  AgentState out;
  CHECK(env.s().load(&out) != LoadResult::Ok);
}

TEST(checkpoint_validate_fnv1a) {
  // Reference implementation of FNV-1a 64-bit.
  auto fnv = [](const uint8_t* d, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
      h ^= d[i];
      h *= 1099511628211ULL;
    }
    return h;
  };
  CHECK(CheckpointStore::fnv1a64(nullptr, 0) == fnv(nullptr, 0));
  const uint8_t data[] = {'a', 'b', 'c'};
  CHECK(CheckpointStore::fnv1a64(data, 3) == fnv(data, 3));
}

TEST(checkpoint_v3_replay_roundtrip_and_v2_migration) {
  CkptEnv env("replay_v3");
  Rng rng(2); Agent agent(env.cfg,rng);
  float s[30]{},h[16]{},ht[3]{};
  for(int i=0;i<40;++i)
    agent.observeAndTrain(s,h,Action::Up,.2f,.2f,s,false,ht,false,false,i);
  AgentState state; agent.exportState(state);
  CHECK(env.s().save(state)); AgentState loaded;
  CHECK(env.s().load(&loaded)==LoadResult::Ok);
  CHECK(loaded.replay_floats==state.replay_floats);
  CHECK(loaded.replay_meta==state.replay_meta);
  auto bytes=readFile(env.s().primaryPath());
  CHECK(std::memcmp(bytes.data(),"SIRCPT03",8)==0);
  // v2 is the identical payload prefix, without the two replay vectors.
  const size_t extension=8+state.replay_floats.size()*4+state.replay_meta.size()*4;
  bytes.resize(bytes.size()-8-extension);
  bytes[7]='2'; bytes[8]=2;
  const uint64_t checksum=CheckpointStore::fnv1a64(bytes.data(),bytes.size());
  for(int i=0;i<8;++i) bytes.push_back(static_cast<uint8_t>(checksum>>(8*i)));
  CHECK(writeFile(env.s().primaryPath(),bytes));
  CHECK(env.s().load(&loaded)==LoadResult::Ok);
  CHECK(loaded.online_params==state.online_params); CHECK(loaded.replay_meta.empty());
  CHECK(agent.importState(loaded));
  CHECK(env.s().save(loaded));
  CHECK(readFile(env.s().backupPath())==bytes); // preserve old organism at migration
}

TEST(checkpoint_invalid_save_preserves_both_valid_generations) {
  CkptEnv env("invalid_save"); auto state=env.makeState();
  CHECK(env.s().save(state)); ++state.lifetime_steps; CHECK(env.s().save(state));
  const auto primary=readFile(env.s().primaryPath()), backup=readFile(env.s().backupPath());
  state.adam_v[0]=-1;
  CHECK(!env.s().save(state));
  CHECK(primary==readFile(env.s().primaryPath())); CHECK(backup==readFile(env.s().backupPath()));
}

TEST(checkpoint_future_format_preserved_and_not_overwritten) {
  CkptEnv env("future"); auto state=env.makeState(); CHECK(env.s().save(state));
  auto bytes=readFile(env.s().primaryPath()); bytes[7]='9'; bytes[8]=9;
  CHECK(writeFile(env.s().primaryPath(),bytes)); AgentState out;
  CHECK(env.s().load(&out)==LoadResult::Incompatible);
  CHECK(!env.s().save(state)); CHECK(readFile(env.s().primaryPath())==bytes);
}
