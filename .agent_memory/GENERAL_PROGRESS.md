# General Progress — Self Improving Rat

Durable, verified progress only. Assumptions are marked.

## 2026-08-04 — Project bootstrap and foundation

- [x] Verified empty project folder, no git; initialized git (local only).
- [x] Verified toolchain: Ubuntu 26.04, cmake 4.2.3, g++ 15.2, pkg-config, make, ninja.
- [x] SDL2 runtime present system-wide; dev package extracted locally to `.deps/` (no root).
- [x] Project skeleton: `include/{app,simulation,learning,rendering,persistence,utility,organism}`,
      `src/...`, `config/`, `data/{checkpoints,logs}`, `.gitignore`, `.agent_memory/`.
- [x] Utility layer: Config, Rng, Logger, RingBuffer, Metrics, 3x5 bitmap font.
- [x] Simulation: Maze (backtracker + braiding), Rat, 30-channel observation, Simulation orchestrator
      with homeostasis integration.
- [x] Organism: homeostasis, development, episodic memory, novelty.
- [x] Learning: GRU, NeuralNet (policy + prediction heads, masks, utility), replay buffer, Agent
      (double-DQN + Adam + prediction + curiosity + episodic + plasticity + consolidation).
- [x] Persistence v2 (atomic writes, backup, validation, legacy handling), rendering, application
      loop, main, CMake, run.sh.

## 2026-08-05 — Verification pass (two concurrent sessions)

- [x] First full compile (nothing had ever been compiled); fixed ~20 compile errors.
- [x] **Bugs found and fixed (all verified by tests):**
  1. `checkpoint.cpp` reader never skipped the 8-byte magic → every checkpoint was rejected as
     corrupt (checkpoint saving had NEVER worked). Added `Reader::skip(8)`.
  2. `checkpoint.cpp` writer omitted the utility-vector count field → reader misread the first
     utility float as the count. Added `b.u32(utility.size())`.
  3. `RingBuffer::push` wrote at `head_` then advanced, while `operator[]`/`sum` treated `head_`
     as the oldest element → **all rolling metrics and every non-full ring read garbage**.
     Fixed push to write at `head_+count_` and advance only when full.
  4. `NeuralNet::updateUtility` indexed the dense-ordered gradient with weight-order indices →
     utility traces (which drive pruning) were computed from misaligned gradients. Fixed to
     resolve dense offsets via `weightToDense`.
  5. `NeuralNet::forward` wrote the prediction head unconditionally; double-DQN target passes
     pass `nullptr` for pred → segfault on first training batch. Added a null guard.
  6. Prediction loss weight `prediction_loss_weight` was never applied; prediction targets used
     the composite reward instead of the external reward (inconsistent with `computeIntrinsics`).
     Fixed: scaled gradient + stored `reward_ext` in replay buffer and episodic memory
     (checkpoint per-entry layout updated to input*2+rnn+3).
  7. Application metrics bugs: revisit metric always false; steps-to-cheese always 1; adaptation
     metric always 0; fps never computed; sim_steps_per_sec counted frames.
- [x] Test suite: `tests/test_framework.h` + `tests/test_main.cpp`, **47 tests, all passing**
      under Release AND ASan+UBSan. Includes GRU/NeuralNet finite-difference gradient checks,
      no-cheating isolation, determinism, checkpoint corruption/recovery, long-run boundedness.
- [x] Build/launch verification: fresh checkout build, damaged-build auto-repair, `--test`,
      SIGINT/SIGTERM graceful save+exit, SIR_MAX_STEPS clean exit, persistence stress
      (14 app-level scenarios), long-run (500k steps: RSS flat at 2.3 MB, 1 thread, 3 fds,
      bounded logs/checkpoints), 300-seed maze sampling (0 failures), render-timing
      determinism (headless vs rendered checkpoints byte-identical except save counters).
- [x] README.md written (accurate to verified behavior; honest about no guaranteed improvement).
- [x] config/default.cfg completed with all organism keys.
- [x] Strict warnings: -Wall -Wextra -Wpedantic -Wshadow, zero warnings; -Wconversion audited
      (only intentional-cast noise, not enabled by default).

## Known limitations (verified)

- The rat found cheese only rarely in long headless runs (e.g., 1 cheese in 500k steps with a
  47x31 maze): exploration is real and unscripted; improvement is not guaranteed.
- XWayland in this environment became wedged after window-close automation; GUI-close
  verification was done by code inspection + the shared shutdown path (verified via signals).
- `load()` returns `Missing` for corrupt-without-backup and incompatible checkpoints (evidence
  files preserved); the `Corrupt`/`Incompatible` LoadResult values are internal to validateFile.
