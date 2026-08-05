#include "persistence/checkpoint.h"

#include "learning/neural_net.h"
#include "simulation/observation.h"
#include "utility/rng.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

namespace sir {

namespace {

// --- little-endian byte buffer writer/reader -------------------------------

class Buf {
 public:
  void u8(uint8_t v) { b_.push_back(v); }
  void u32(uint32_t v) {
    for (int i = 0; i < 4; ++i) b_.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
  }
  void u64(uint64_t v) {
    for (int i = 0; i < 8; ++i) b_.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
  }
  void i64(int64_t v) { u64(static_cast<uint64_t>(v)); }
  void f32(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    u32(bits);
  }
  void bytes(const void* p, size_t n) {
    if (n == 0) return;
    const uint8_t* q = static_cast<const uint8_t*>(p);
    b_.insert(b_.end(), q, q + n);
  }
  void f32_vec(const std::vector<float>& v) { bytes(v.data(), v.size() * 4); }
  void u32_vec(const std::vector<uint32_t>& v) { bytes(v.data(), v.size() * 4); }
  const std::vector<uint8_t>& data() const { return b_; }

 private:
  std::vector<uint8_t> b_;
};

class Reader {
 public:
  Reader(const uint8_t* p, size_t n) : p_(p), n_(n) {}
  bool u32(uint32_t* v) {
    if (pos_ + 4 > n_) return false;
    *v = static_cast<uint32_t>(p_[pos_]) | (static_cast<uint32_t>(p_[pos_ + 1]) << 8) |
         (static_cast<uint32_t>(p_[pos_ + 2]) << 16) |
         (static_cast<uint32_t>(p_[pos_ + 3]) << 24);
    pos_ += 4;
    return true;
  }
  bool u64(uint64_t* v) {
    if (pos_ + 8 > n_) return false;
    uint64_t r = 0;
    for (int i = 0; i < 8; ++i) r |= static_cast<uint64_t>(p_[pos_ + i]) << (8 * i);
    *v = r;
    pos_ += 8;
    return true;
  }
  bool i64(int64_t* v) { return u64(reinterpret_cast<uint64_t*>(v)); }
  bool f32(float* v) {
    uint32_t bits;
    if (!u32(&bits)) return false;
    std::memcpy(v, &bits, 4);
    return true;
  }
  bool bytes(void* out, size_t n) {
    if (pos_ + n > n_) return false;
    std::memcpy(out, p_ + pos_, n);
    pos_ += n;
    return true;
  }
  bool skip(size_t n) {
    if (pos_ + n > n_) return false;
    pos_ += n;
    return true;
  }
  size_t pos() const { return pos_; }
  size_t size() const { return n_; }

 private:
  const uint8_t* p_;
  size_t n_;
  size_t pos_ = 0;
};

constexpr char kMagic[] = "SIRCPT";
constexpr uint32_t kFormatVersion = 2;
constexpr size_t kMaxFileBytes = 64 * 1024 * 1024;  // sanity cap

bool fileExists(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool copyFile(const std::string& from, const std::string& to) {
  std::ifstream in(from, std::ios::binary);
  std::ofstream out(to, std::ios::binary | std::ios::trunc);
  if (!in.is_open() || !out.is_open()) return false;
  out << in.rdbuf();
  out.flush();
  return out.good();
}

// Reads the first 8 bytes; legacy v1 files use magic "SIRCPT01".
bool isLegacyV1(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) return false;
  char m[8] = {0};
  in.read(m, 8);
  return in.gcount() == 8 && std::memcmp(m, "SIRCPT01", 8) == 0;
}

}  // namespace

uint64_t CheckpointStore::fnv1a64(const uint8_t* data, size_t len) {
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < len; ++i) {
    h ^= data[i];
    h *= 1099511628211ULL;
  }
  return h;
}

CheckpointStore::CheckpointStore(const Config& cfg, Logger& log)
    : cfg_(cfg), log_(log) {
  primary_ = cfg.checkpoint_dir + "/rat.sir";
  backup_ = cfg.checkpoint_dir + "/rat.sir.bak";
}

// --- serialization ---------------------------------------------------------

namespace {

std::vector<uint8_t> serializeState(const AgentState& s) {
  Buf b;
  b.bytes(kMagic, 6);
  b.bytes("02", 2);
  b.u32(kFormatVersion);
  b.u32(static_cast<uint32_t>(s.input));
  b.u32(static_cast<uint32_t>(s.rnn));
  b.u32(static_cast<uint32_t>(s.policy_out));
  b.u32(static_cast<uint32_t>(s.pred_out));
  b.f32(s.lr);
  b.f32(s.gamma);
  b.f32(s.tau);
  b.u64(s.training_steps);
  b.u64(s.lifetime_steps);
  b.u64(s.cheese_total);
  b.u32(s.maze_generations);
  b.u32(s.checkpoints_saved);
  b.u32(s.invalid_updates);
  b.f32(s.epsilon);
  b.u32(s.seed);
  b.i64(s.timestamp_utc);
  b.u32(static_cast<uint32_t>(s.rng_state.size()));
  b.bytes(s.rng_state.data(), s.rng_state.size());

  // Organism internal state.
  const Homeostasis::Snapshot& h = s.homeo;
  b.f32(h.energy);
  b.f32(h.hunger);
  b.f32(h.fatigue);
  b.f32(h.stress);
  b.f32(h.curiosity_need);
  b.f32(h.satisfaction);
  b.f32(h.uncertainty);
  b.f32(s.novelty_ema);
  b.f32(s.pred_loss_ema);
  b.f32(s.uncertainty);

  // Lifetime age / runtime / consolidation history.
  b.u64(s.active_runtime_ms);
  b.u32(s.consolidation_cycles);
  b.u64(s.last_consolidation_step);
  b.u64(s.consolidation_train_ops_total);
  b.u64(s.explored_count);
  b.u32(s.structural_accepted);
  b.u32(s.structural_rejected);
  b.u64(s.pruned_total);
  b.u64(s.rewired_total);

  // Episodic memory.
  const size_t per_entry_floats =
      static_cast<size_t>(s.input) * 2 + s.rnn + 3;
  b.u32(static_cast<uint32_t>(s.episodic_capacity));
  b.u32(static_cast<uint32_t>(s.episodic_floats.size() / per_entry_floats));
  b.u32(static_cast<uint32_t>(s.episodic_floats.size()));
  b.f32_vec(s.episodic_floats);
  b.u32(static_cast<uint32_t>(s.episodic_meta.size()));
  b.u32_vec(s.episodic_meta);

  // Novelty table.
  b.u32(static_cast<uint32_t>(s.novelty_table.size()));
  b.u32_vec(s.novelty_table);

  // Neural parameters.
  b.u32(static_cast<uint32_t>(s.online_params.size()));
  b.f32_vec(s.online_params);
  b.f32_vec(s.target_params);
  b.f32_vec(s.adam_m);
  b.f32_vec(s.adam_v);
  b.u32(static_cast<uint32_t>(s.masks.size()));
  b.bytes(s.masks.data(), s.masks.size());
  // Utility traces: count + data (the reader validates the count against the
  // expected parameter count; without the count field the first four float
  // bytes would be misread as the count).
  b.u32(static_cast<uint32_t>(s.utility.size()));
  b.f32_vec(s.utility);

  // Checksum over everything written so far.
  const uint64_t checksum = CheckpointStore::fnv1a64(b.data().data(), b.data().size());
  b.u64(checksum);
  return b.data();
}

// Fills *out from a v2-layout byte stream; returns LoadResult.
LoadResult deserializeState(const std::vector<uint8_t>& bytes, const Config& cfg,
                            AgentState* out) {
  if (bytes.size() < 96) return LoadResult::Corrupt;
  if (bytes.size() > kMaxFileBytes) return LoadResult::Corrupt;
  if (bytes.size() < 8 || std::memcmp(bytes.data(), "SIRCPT02", 8) != 0) return LoadResult::Corrupt;

  Reader r(bytes.data(), bytes.size());
  // Skip the 8-byte magic before reading the version field (the writer
  // serializes magic first). Without this, the first 4 magic bytes are
  // interpreted as the version and every file is rejected as corrupt.
  if (!r.skip(8)) return LoadResult::Corrupt;
  uint32_t version;
  if (!r.u32(&version)) return LoadResult::Corrupt;
  if (version > kFormatVersion) return LoadResult::Corrupt;  // newer than us

  AgentState s;
  uint32_t v;
  if (!r.u32(&v)) return LoadResult::Corrupt;
  s.input = static_cast<int>(v);
  if (!r.u32(&v)) return LoadResult::Corrupt;
  s.rnn = static_cast<int>(v);
  if (!r.u32(&v)) return LoadResult::Corrupt;
  s.policy_out = static_cast<int>(v);
  if (!r.u32(&v)) return LoadResult::Corrupt;
  s.pred_out = static_cast<int>(v);
  if (!r.f32(&s.lr) || !r.f32(&s.gamma) || !r.f32(&s.tau)) return LoadResult::Corrupt;
  if (!r.u64(&s.training_steps) || !r.u64(&s.lifetime_steps) ||
      !r.u64(&s.cheese_total)) return LoadResult::Corrupt;
  if (!r.u32(&s.maze_generations) || !r.u32(&s.checkpoints_saved) ||
      !r.u32(&s.invalid_updates)) return LoadResult::Corrupt;
  if (!r.f32(&s.epsilon) || !r.u32(&s.seed) || !r.i64(&s.timestamp_utc)) return LoadResult::Corrupt;
  uint32_t rng_len;
  if (!r.u32(&rng_len)) return LoadResult::Corrupt;
  if (rng_len > 65536) return LoadResult::Corrupt;
  s.rng_state.resize(rng_len);
  if (rng_len > 0 && !r.bytes(s.rng_state.data(), rng_len)) return LoadResult::Corrupt;

  if (!r.f32(&s.homeo.energy) || !r.f32(&s.homeo.hunger) ||
      !r.f32(&s.homeo.fatigue) || !r.f32(&s.homeo.stress) ||
      !r.f32(&s.homeo.curiosity_need) || !r.f32(&s.homeo.satisfaction) ||
      !r.f32(&s.homeo.uncertainty)) return LoadResult::Corrupt;
  if (!r.f32(&s.novelty_ema) || !r.f32(&s.pred_loss_ema) || !r.f32(&s.uncertainty)) return LoadResult::Corrupt;

  if (!r.u64(&s.active_runtime_ms)) return LoadResult::Corrupt;
  if (!r.u32(&s.consolidation_cycles)) return LoadResult::Corrupt;
  if (!r.u64(&s.last_consolidation_step)) return LoadResult::Corrupt;
  if (!r.u64(&s.consolidation_train_ops_total)) return LoadResult::Corrupt;
  if (!r.u64(&s.explored_count)) return LoadResult::Corrupt;
  if (!r.u32(&s.structural_accepted) || !r.u32(&s.structural_rejected)) return LoadResult::Corrupt;
  if (!r.u64(&s.pruned_total) || !r.u64(&s.rewired_total)) return LoadResult::Corrupt;

  // Topology must match the current configuration.
  const int expected_input = kObservationBase * cfg.observation_frames;
  if (s.input != expected_input || s.rnn != cfg.rnn_hidden ||
      s.policy_out != 4 || s.pred_out != 16) {
    return LoadResult::Incompatible;
  }
  const size_t per_entry_floats = static_cast<size_t>(s.input) * 2 + s.rnn + 3;

  uint32_t ep_cap, ep_count, ep_floats_n, ep_meta_n;
  if (!r.u32(&ep_cap) || !r.u32(&ep_count) || !r.u32(&ep_floats_n)) return LoadResult::Corrupt;
  if (ep_count > ep_cap || ep_floats_n != ep_count * per_entry_floats) return LoadResult::Corrupt;
  s.episodic_capacity = ep_cap;
  s.episodic_floats.resize(ep_floats_n);
  if (!r.bytes(s.episodic_floats.data(), ep_floats_n * 4)) return LoadResult::Corrupt;
  if (!r.u32(&ep_meta_n)) return LoadResult::Corrupt;
  if (ep_meta_n != ep_count * 4) return LoadResult::Corrupt;
  s.episodic_meta.resize(ep_meta_n);
  if (!r.bytes(s.episodic_meta.data(), ep_meta_n * 4)) return LoadResult::Corrupt;

  uint32_t nov_n;
  if (!r.u32(&nov_n)) return LoadResult::Corrupt;
  if (nov_n % 2 != 0 || nov_n > 1u << 20) return LoadResult::Corrupt;
  s.novelty_table.resize(nov_n);
  if (!r.bytes(s.novelty_table.data(), nov_n * 4)) return LoadResult::Corrupt;

  uint32_t pcount;
  if (!r.u32(&pcount)) return LoadResult::Corrupt;
  const size_t expected =
      static_cast<size_t>(NeuralNet::paramCount(s.input, s.rnn, s.policy_out, s.pred_out));
  if (pcount != expected) return LoadResult::Corrupt;
  s.online_params.resize(pcount);
  s.target_params.resize(pcount);
  s.adam_m.resize(pcount);
  s.adam_v.resize(pcount);
  if (!r.bytes(s.online_params.data(), pcount * 4) ||
      !r.bytes(s.target_params.data(), pcount * 4) ||
      !r.bytes(s.adam_m.data(), pcount * 4) ||
      !r.bytes(s.adam_v.data(), pcount * 4)) return LoadResult::Corrupt;
  uint32_t mask_bytes;
  if (!r.u32(&mask_bytes)) return LoadResult::Corrupt;
  if (mask_bytes != (expected + 7) / 8) return LoadResult::Corrupt;
  s.masks.resize(mask_bytes);
  if (!r.bytes(s.masks.data(), mask_bytes)) return LoadResult::Corrupt;
  uint32_t util_n;
  if (!r.u32(&util_n)) return LoadResult::Corrupt;
  if (util_n != expected) return LoadResult::Corrupt;
  s.utility.resize(util_n);
  if (!r.bytes(s.utility.data(), util_n * 4)) return LoadResult::Corrupt;

  // Checksum covers everything up to this point.
  const uint64_t stored_checksum_pos = r.pos();
  uint64_t stored;
  if (!r.u64(&stored)) return LoadResult::Corrupt;
  if (r.pos() != bytes.size()) return LoadResult::Corrupt;  // trailing garbage
  const uint64_t computed =
      CheckpointStore::fnv1a64(bytes.data(), stored_checksum_pos);
  if (computed != stored) return LoadResult::Corrupt;

  if (!s.allFinite()) return LoadResult::Corrupt;
  // RNG state must round-trip (defensive).
  Rng probe(s.seed);
  if (!s.rng_state.empty() && !probe.restoreState(s.rng_state)) return LoadResult::Corrupt;

  *out = std::move(s);
  return LoadResult::Ok;
}

}  // namespace

LoadResult CheckpointStore::validateFile(const std::string& path,
                                         const Config& cfg, AgentState* out,
                                         Logger& log) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in.is_open()) return LoadResult::Missing;
  const std::streamoff size = in.tellg();
  if (size < 96 || size > static_cast<std::streamoff>(kMaxFileBytes)) {
    log.warn("checkpoint file rejected: bad size " + path);
    return LoadResult::Corrupt;
  }
  in.seekg(0);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  if (!in.read(reinterpret_cast<char*>(bytes.data()), size)) {
    log.warn("checkpoint file rejected: read error " + path);
    return LoadResult::Corrupt;
  }
  return deserializeState(bytes, cfg, out);
}

bool CheckpointStore::writeAtomic(const AgentState& state) {
  const std::vector<uint8_t> bytes = serializeState(state);
  const std::string tmp = primary_ + ".tmp";

  FILE* f = fopen(tmp.c_str(), "wb");
  if (!f) {
    log_.error("cannot open checkpoint temp file: " + tmp);
    return false;
  }
  const size_t written = fwrite(bytes.data(), 1, bytes.size(), f);
  if (written != bytes.size() || fflush(f) != 0 ||
      fsync(fileno(f)) != 0 || fclose(f) != 0) {
    log_.error("checkpoint write failed (disk full?): " + tmp);
    ::remove(tmp.c_str());
    return false;
  }

  // Promote the current valid checkpoint to backup, then install the new one.
  if (fileExists(primary_)) {
    ::remove(backup_.c_str());
    if (::rename(primary_.c_str(), backup_.c_str()) != 0) {
      log_.error("checkpoint backup rename failed");
      ::remove(tmp.c_str());
      return false;
    }
  }
  if (::rename(tmp.c_str(), primary_.c_str()) != 0) {
    log_.error("checkpoint install rename failed");
    ::remove(tmp.c_str());
    // Restore the backup if we moved it away.
    if (fileExists(backup_)) ::rename(backup_.c_str(), primary_.c_str());
    return false;
  }

  // Validate what we just wrote; never leave an invalid primary in place.
  AgentState check;
  const LoadResult r = validateFile(primary_, cfg_, &check, log_);
  if (r != LoadResult::Ok) {
    log_.error("post-write validation failed; restoring backup");
    if (fileExists(backup_)) ::rename(backup_.c_str(), primary_.c_str());
    else ::remove(primary_.c_str());
    return false;
  }
  return true;
}

bool CheckpointStore::save(const AgentState& state) {
  if (!writeAtomic(state)) return false;
  ++checkpoints_saved_;
  return true;
}

LoadResult CheckpointStore::load(AgentState* out) {
  if (fileExists(primary_)) {
    const LoadResult r = validateFile(primary_, cfg_, out, log_);
    if (r == LoadResult::Ok) {
      log_.info("checkpoint loaded: " + primary_);
      return LoadResult::Ok;
    }
    if (r == LoadResult::Incompatible) {
      log_.warn("checkpoint topology incompatible with current config: " + primary_);
    } else {
      log_.warn("checkpoint corrupt; moving aside: " + primary_);
      const std::string suffix = isLegacyV1(primary_) ? ".legacy-v1" : ".bad";
      ::rename(primary_.c_str(), (primary_ + suffix).c_str());
    }
  }
  if (fileExists(backup_)) {
    const LoadResult r = validateFile(backup_, cfg_, out, log_);
    if (r == LoadResult::Ok) {
      log_.warn("recovered from backup checkpoint: " + backup_);
      // Restore a primary copy so subsequent loads start from the primary.
      if (!fileExists(primary_)) copyFile(backup_, primary_);
      return LoadResult::Ok;
    }
    if (r == LoadResult::Incompatible) {
      log_.warn("backup checkpoint topology incompatible: " + backup_);
    } else {
      log_.warn("backup checkpoint corrupt; moving aside: " + backup_);
      const std::string suffix = isLegacyV1(backup_) ? ".legacy-v1" : ".bad";
      ::rename(backup_.c_str(), (backup_ + suffix).c_str());
    }
  }
  return LoadResult::Missing;
}

}  // namespace sir
