#pragma once
// Checkpoint persistence. Binary format v2 ("SIRCPT02") stores the complete
// organism state (topology, hyperparameters, counters, homeostasis, age,
// runtime, recurrent network + heads, Adam moments, masks, utility traces,
// episodic memory, novelty table, RNG state) with an FNV-1a64 checksum.
//
// Safety properties:
//  - atomic writes: temp file + fsync + rename; previous checkpoint kept as
//    .bak (the newest valid checkpoint is never overwritten by partial data)
//  - files are validated after writing (re-read + checksum)
//  - load order: primary -> backup -> fresh organism with a clear log
//  - legacy v1 files ("SIRCPT01", pre-recurrent layout) cannot be mapped:
//    they are preserved with a .legacy suffix and a fresh organism starts
//  - an old format version with a compatible layout is migrated on load

#include "learning/agent_state.h"
#include "utility/config.h"
#include "utility/logger.h"

#include <cstdint>
#include <string>
#include <vector>

namespace sir {

enum class LoadResult {
  Ok,
  Missing,        // no file found
  Corrupt,        // file failed validation (checksum/size/NaN)
  Incompatible,   // topology does not match the current configuration
};

class CheckpointStore {
 public:
  CheckpointStore(const Config& cfg, Logger& log);

  // Atomic save; keeps a backup of the previous valid checkpoint. Returns
  // false if the write or the post-write validation failed.
  bool save(const AgentState& state);

  // Loads the newest valid checkpoint (primary, then backup). On corrupt
  // primary + valid backup, the primary is renamed to .bad and recovery is
  // logged. Returns Missing when nothing loadable exists.
  LoadResult load(AgentState* out);

  std::string primaryPath() const { return primary_; }
  std::string backupPath() const { return backup_; }
  uint64_t checkpointsSaved() const { return checkpoints_saved_; }

  // Checksum (FNV-1a 64-bit).
  static uint64_t fnv1a64(const uint8_t* data, size_t len);

  // Validates a file and fills *out on success.
  static LoadResult validateFile(const std::string& path, const Config& cfg,
                                 AgentState* out, Logger& log);

 private:
  bool writeAtomic(const AgentState& state);
  Config cfg_;
  Logger& log_;
  std::string primary_;
  std::string backup_;
  uint64_t checkpoints_saved_ = 0;
};

}  // namespace sir
