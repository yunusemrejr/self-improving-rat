# HANDOFF — Self Improving Rat (master handoff, 2026-08-04)

> **STATUS UPDATE (2026-08-05, verification pass):** Everything in the
> "NOT YET WRITTEN" / "NEXT SESSION" lists below is now implemented and
> verified: CMakeLists.txt, run.sh, src/main.cpp, tests/ (47 tests, all
> passing under Release and ASan+UBSan), README.md, and config/default.cfg
> (all organism keys). The design details below remain accurate except:
> the observation vector is now **30 channels** (28 + 2 proprioception
> position channels added during verification), the episodic per-entry
> layout is now `input*2 + rnn + 3` (reward_ext added), and several bugs
> were fixed (checkpoint magic-skip + utility count serialization,
> RingBuffer push/index convention, utility-trace gradient indexing,
> null-pred guard, prediction-loss weight, metrics). See
> `.agent_memory/GENERAL_PROGRESS.md` for the verified record. First commit:
> `370a58f`.

**Project root:** `/home/yemre/Desktop/self-improving-rat`
**Objective:** Ubuntu-native C++20/CMake/SDL2 desktop app: a persistent artificial-life
rat (RL + homeostasis + world model + recurrent memory + episodic memory + plasticity +
consolidation) navigating procedurally generated mazes, finding cheese, persisting its
developmental state. Single entry point `./run.sh`. Base spec + artificial-life follow-up
spec are BOTH in force. No GPU, no ML frameworks, no network, tiny resource footprint.

---

## 0. WHERE WE ARE (summary)

- ✅ Environment verified; git initialized (local only, no remote); no commits yet.
- ✅ `config/default.cfg` written BUT **OUT OF DATE** — needs all organism keys appended
  (Config struct has the keys with defaults, so the app works without them, but the file
  must document them; copy the key list from `include/utility/config.h`).
- ✅ Utility layer: `config.{h,cpp}`, `rng.{h,cpp}`, `logger.{h,cpp}`, `ring_buffer.h`,
  `rolling_stats.{h,cpp}`, `bitmap_font.{h,cpp}`.
- ✅ Organism module: `homeostasis.{h,cpp}`, `development.{h,cpp}`,
  `episodic_memory.{h,cpp}`, `novelty.{h,cpp}`.
- ✅ Simulation module: `rat.h`, `maze.{h,cpp}`, `observation.{h,cpp}` (28 channels),
  `simulation.{h,cpp}` (homeostasis integrated).
- ✅ Learning module: `gru.{h,cpp}`, `neural_net.{h,cpp}` (GRU + policy/pred heads +
  masks + utility), `replay_buffer.{h,cpp}`, `agent_state.h`, `agent.{h,cpp}` (DQN +
  prediction + curiosity + episodic + plasticity + consolidation ops).
- ✅ Persistence: `checkpoint.{h,cpp}` (v2 format, atomic writes, backup, validation,
  legacy-v1 handling).
- ✅ Rendering: `renderer.{h,cpp}` (SDL2 monochrome, 3x5 font, panel, resting visual).
- ❌ **NOT YET WRITTEN**: `src/app/application.{h,cpp}`, `src/main.cpp`, `CMakeLists.txt`,
  `run.sh`, `tests/`, `README.md`. **NOTHING HAS BEEN COMPILED YET** — expect compile
  errors to fix (especially agent.cpp / checkpoint.cpp / renderer.cpp, the newest files).

---

## 1. ENVIRONMENT FACTS (verified)

- Ubuntu 26.04 LTS, 12 cores, cmake 4.2.3, g++ 15.2 (C++20 OK), pkg-config 2.5.1,
  make, ninja. `sudo` needs a password (NOT available to agents).
- **SDL2 runtime** `libSDL2-2.0.so.0` IS installed system-wide (2.32.10).
- **SDL2 dev is NOT installed** and cannot be apt-installed without the user's password.
- Workaround in place: `libsdl2-dev_2.32.10+dfsg-6_amd64.deb` downloaded and extracted to
  `.deps/extracted/` (gitignored). Headers: `.deps/extracted/usr/include/SDL2/SDL.h`.
  Libs: `.deps/extracted/usr/lib/x86_64-linux-gnu/{libSDL2.so,libSDL2.a,...}`.
- ⚠️ Its `sdl2.pc` hardcodes `prefix=/usr` → pkg-config is USELESS for the local build.
  **CMake must support an explicit prefix fallback**: `SIR_SDL2_PREFIX` (cmake var or
  env) → include `${SIR_SDL2_PREFIX}/usr/include`, lib
  `${SIR_SDL2_PREFIX}/usr/lib/x86_64-linux-gnu` (detect arch dir via
  `CMAKE_SYSTEM_PROCESSOR` or try `x86_64-linux-gnu` then `aarch64-linux-gnu`).
  Order of preference in CMake: 1) system pkg-config sdl2, 2) SIR_SDL2_PREFIX, 3) error.
  Runtime lib resolves system-wide → no LD_LIBRARY_PATH needed.
- For END USERS, `run.sh` must detect missing `libsdl2-dev` and offer
  `sudo apt-get install -y libsdl2-dev` via a `--install-deps` flag (never auto-install
  without the flag; detection = pkg-config --exists sdl2).

## 2. HOW TO BUILD / VERIFY (commands for the next session)

```bash
cd /home/yemre/Desktop/self-improving-rat
# local build with extracted SDL2 (no root):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSIR_SDL2_PREFIX=$PWD/.deps/extracted
cmake --build build -j12
./build/sir_tests                 # tests (no SDL needed)
# headless GUI smoke (dummy video driver):
SDL_VIDEODRIVER=dummy SIR_MAX_STEPS=300 SIR_SCREENSHOT=/tmp/rat.bmp ./build/sir
# visual check: read /tmp/rat.bmp with the image reader
# deterministic long run:
SDL_VIDEODRIVER=dummy SIR_MAX_STEPS=2000 ./build/sir   # then check data/checkpoints/rat.sir
```
Env hooks (must be implemented in main.cpp/application.cpp): `SIR_SEED`, `SIR_HEADLESS=1`
(no SDL at all), `SIR_MAX_STEPS` (clean exit after N sim steps), `SIR_SCREENSHOT=path.bmp`
(save one frame after ~30 rendered frames, then continue), `SIR_CONFIG=path`.

## 3. ARCHITECTURE & DATA FLOW (decisions, assumed unless marked Known)

Namespaces `sir`; dirs `include/{app,simulation,learning,rendering,persistence,utility,organism}`.

**Per-sim-step flow (Application must implement exactly this):**
1. `s = sim.observe()` (const float*, size = input_size_ = 28 * observation_frames).
2. `d = agent.selectAction(s)` → advances agent's internal recurrent state `h_`,
   fills `last_prediction()` (16 sigmoid outputs) for this (s, h_prev).
3. `out = sim.step(d.action)` → returns `StepOutcome{reward_nav, cheese_reached,
   maze_regenerated, wall_hit, moved, homeo_before, homeo_after}`.
4. `agent.computeIntrinsics(s, s2, homeo_targets, reward_ext)` where
   `homeo_targets[k] = homeo_after[k] - homeo_before[k]` (raw deltas for
   energy/hunger/stress, in [-1,1]) and `reward_ext = out.reward_nav + homeostatic_reward`
   (the externally determined part; the model predicts THIS, not the intrinsic bonuses).
   Sets: `lastNovelty()`, `lastCuriosityReward()` (= gain * novelty, capped 0.02),
   `lastPredError()`, `uncertainty()` EMA, `predictionLossEma()`.
5. Compose total reward in the app:
   `r_total = out.reward_nav` (cheese +10 / wall −0.2 / step −0.02 / revisit −0.05)
   `+ homeo_reward(homeo_after)` where
   `homeo_reward = −energy_factor*(1−E) − hunger_factor*H − fatigue_factor*F
                   − stress_factor*S² + satisfaction_factor*(Sat−0.5)`
   (factors from config `homeo_reward_*`, all bounded)
   `+ curiosity_reward * (0.5 + 0.5 * curiosity_need)`  (curiosity modulated by the need)
   `+ prediction_reward_gain * (1 − uncertainty)`
   `+ collapse_penalty` (only when action entropy over last 200 steps < 0.2 AND
   recent avg reward < −0.05; contextual anti-collapse, documented)
6. `agent.observeAndTrain(s, h_prev, d.action, r_total, s2, out.cheese_reached,
   homeo_targets, out.cheese_reached, out.wall_hit, sim.lifetimeSteps())`.
   Note: `h_prev` = the recurrent state BEFORE selectAction (agent keeps it internally;
   expose `recurrentState()` — read it BEFORE selectAction or keep a copy; simplest:
   `const float* h_prev = agent.recurrentState();` BEFORE calling selectAction).
7. `sim.setNoveltySignal(agent.lastNovelty())`, `sim.homeostasis().applyNovelty(...)`,
   `sim.homeostasis().setUncertainty(agent.uncertainty())` — before the next observe.
8. Metrics: `metrics.recordStep(r_total, wall_hit, revisit, action, prev_action,
   explored, homeo_after_snapshot, novelty, curiosity_reward, pred_loss)`.
   Cheese → `metrics.recordCheese(steps_since_cheese_before)`; regen → reset an
   `adaptation_start` counter and `metrics.recordAdaptation(steps)` at next cheese.

**Consolidation controller (Application):**
- Triggers (checked every step, all bounded by a min interval):
  cheese count since last consolidation ≥ `consolidation_every_cheeses` (4), OR
  `fatigue ≥ consolidation_fatigue_threshold` (0.9), OR maze regenerated this step;
  cooldown: `steps_since_last_consolidation ≥ consolidation_min_interval_steps` (2000).
- During consolidation (max `consolidation_max_steps` = 120 sim steps, ops capped at
  `consolidation_max_train_ops` = 60):
  - no rat movement; each tick: `homeostasis.update(Events{resting=true})` (fatigue
    recovery), `agent.consolidationTrainOps(n)` spread across ticks (e.g., 1 op per
    2 ticks so CPU stays flat), at the end `agent.plasticityEvaluate(eval_obs,
    cfg.plasticity_eval_passes)` (eval_obs = ring of last ~64 observations kept by app),
    then `saveCheckpointNow("consolidation")`.
  - `sim.setResting(true)` during, `false` after. `agent.resetRecurrent()` after
    consolidation (lifecycle boundary) — also after maze regeneration.
  - Track `cheeses_since_last_consolidation`, `last_consolidation_step`,
    `agent.consolidation_cycles++` (agent owns the counter; add a public method or
    increment via a new Agent::noteConsolidationCycle() — agent.h currently exposes
    consolidationCycles() read-only; the app must be able to increment → **add
    `void noteConsolidationCycle()` to Agent** or handle the counter in the app and
    restore into AgentState at save time (AgentState has consolidation_cycles; app can
    set state.consolidation_cycles directly before save — simpler, no code change).
    Decision: app owns the cycle count and writes it into AgentState at save.
- GUI: rat dims/curls when consolidating (renderer handles via PanelData.consolidating).

**Determinism:** single-threaded; RNG draw order must be stable: maze gen → cheese
placement → rat start → exploration draws → batch sampling → plasticity. Never add
non-deterministic sources (no random_device after seed; time only affects pacing).
Tests assert identical trajectories + identical serialized checkpoints for same seed.

## 4. OBSERVATION VECTOR (28 channels; kObservationBase=28 in observation.h)

```
[0..7]   wall bits N,NE,E,SE,S,SW,W,NW (1=blocked)
[8..11]  cheese scent N,E,S,W ∈ [0,1]: max over cheeses of
         (1 − dist/(radius+1)) * max(0, dot(dir, offset)/manhattan)  [distance-only, walls do not block smell]
[12..15] last action one-hot U,D,L,R
[16]     wall-hit flag of last action
[17]     revisit signal = max(0, 1 − steps_since_last_entry/8)
[18]     last reward mapped: (clamp(r,−1,1)+1)/2
[19..25] homeostasis: energy, hunger, fatigue, stress, curiosity_need, satisfaction, uncertainty
[26]     recent novelty signal
[27]     resting flag (1 during consolidation)
```
Input size = 28 * `observation_frames` (default 1; the GRU provides temporal memory;
stacking remains configurable 1..4).

## 5. LEARNING SYSTEM (implemented; verify by building)

- **GRU** (16 hidden; `learning/gru.{h,cpp}`): z,r gates + tanh candidate; single-step
  forward/backward; h_prev treated as constant (truncated BPTT-1). Param layout
  `[Wz|Uz|bz|Wr|Ur|br|Wh|Uh|bh]`, row-major `Wg[i*h+j]` (input i → gate j),
  `Ug[j*h+k]` (recurrent j → gate k). Backward recomputes forward internally.
  ⚠️ `GruLayer::forward` REQUIRES `h_out` ≠ `h_prev` (no aliasing) — agent handles this.
- **NeuralNet** (`learning/neural_net.{h,cpp}`): GRU → policy head (Wq,bq → 4 Q-values)
  + prediction head (Wp,bp → 16 sigmoid outputs). Dense param layout
  `[gru | Wq | bq | Wp | bp]`; `paramCount = GruLayer::paramCount + rnn*4+4 + rnn*16+16`.
  Masks/utility/init-scales cover **weights only** (bias bits never set in mask).
  Mask index order: `[Wz|Uz|Wr|Ur|Wh|Uh|Wq|Wp]`; verified mapping to dense offsets:
  Wz:+0, Uz:+0, Wr:+rnn, Ur:+rnn, Wh:+2rnn, Uh:+2rnn, Wq:+3rnn, Wp:+3rnn+po
  (see `weightToDense`). `backward` applies the sigmoid chain for the pred head
  internally (pass dL/d(pred_output), not pre-activation).
  With input=28, rnn=16, po=4, pr=16: params = 2160 + 64+4 + 256+16 = 2500.
- **Replay buffer**: 8192 transitions `{s(28), s2(28), h_prev(16), action u8, reward f32,
  done u8, homeo_targets(3 f32)}` ≈ 2.5 MB, ring, preallocated.
- **Agent** (`learning/agent.{h,cpp}`): double-DQN (a* from online net on (s2,h_prev),
  value from target net on (s2,h_prev); stale-state approximation), Adam
  (β1=.9, β2=.999, eps 1e-8, bias-corrected, per-batch step), global grad clip 1.0,
  soft target tau 0.02 (target net starts as copy of online), utility trace
  `u = 0.999*u + 0.001*|g*w|` (active weights only), train every 4 steps, batch 32,
  epsilon schedule: `end + (start−end)*max(0,1−steps/decay)` × `epsilonScale(age)`,
  floor = exploration_end. NaN guard: pre-update snapshot; on invalid → restore params,
  zero moments, `invalid_updates++`. Prediction loss weight `prediction_loss_weight`
  (0.3) — **TODO: the prediction gradient in trainBatchOn is currently added with
  weight 1.0; multiply `grad_p` by the configured weight inside trainBatchOn** (or scale
  grad_p before backward; decision: scale grad_p by cfg_.prediction_loss_weight).
- **Episodic memory**: capacity 256; significance: cheese 3.0, |r|>1 → 1.5,
  novelty>0.6 → 1.0, pred_err>0.3 → 0.8, else 0.1 (insert only ≥ 0.3); full →
  replace lowest significance (ties: oldest); `done` flag for replay = `kFlagCheese`.
- **Novelty**: quantized 30-bit observation code (see novelty.cpp `code()`), linear-
  probing hash table (capacity 1024, counts halved when full), novelty = 1/√count.
  Blended: `0.5*state + 0.5*pred_error` (normalized: min(1, mse/0.25)).
- **Plasticity** (only in `plasticityEvaluate`, called from consolidation): prune weakest
  actives (utility < median × 0.05, cap = 4×rewire cap, never below
  max_active_fraction × weight_bits), reactivate up to `rewire_per_consolidation ×
  plasticityScale(age)` dormant with highest utility (fresh weight ±0.5×init_scale),
  strengthen top 5% actives ×(1+strengthen_factor); bounded eval (finite, |q|≤100 on
  recent obs); rollback + rejected++ on failure; else accepted++.

## 6. HOMESTASIS (organism/homeostasis.{h,cpp}) — all in [0,1], deterministic

- Per step: energy −= cost_step (+ cost_move if moved; +0.35 on cheese);
  hunger += rate (−0.5 on cheese); fatigue += per_move (resting: −recovery_rest;
  idle: −recovery_idle); stress: wall → +wall_hit (+repeat_bonus if streak ≥ 2,
  streak resets on move), moved → −success_relief, cheese → −cheese_relief,
  always −recovery_rate; curiosity_need += rate, −= reduce × novelty (via
  `applyNovelty`); satisfaction: cheese +0.4 else reverts to 0.5 at reversion rate.
- `setUncertainty(u)` feeds the prediction-uncertainty channel. `stressFreeSteps()`
  counts steps since last stress increase (transient).
- Defaults in `HomeoParams`; config keys mirror them (energy_cost_step, ..., all in
  config/default.cfg section "[organism] homeostasis").

## 7. CHECKPOINT v2 (persistence/checkpoint.{h,cpp}) — exact layout

Magic `"SIRCPT02"` (8 bytes incl. version digits) + u32 version=2, then in order:
input, rnn, policy_out, pred_out (u32); lr, gamma, tau (f32); training_steps,
lifetime_steps, cheese_total (u64); maze_generations, checkpoints_saved,
invalid_updates (u32); epsilon (f32); seed (u32); timestamp (i64); rng_state_len (u32)
+ bytes; homeostasis snapshot 7×f32; novelty_ema, pred_loss_ema, uncertainty (f32);
active_runtime_ms (u64); consolidation_cycles (u32); last_consolidation_step (u64);
consolidation_train_ops_total (u64); explored_count (u64); structural_accepted,
structural_rejected (u32); pruned_total, rewired_total (u64); episodic_capacity (u32),
episodic_count (u32), episodic float count (u32) + floats [s|s2|h|reward|sig per entry],
episodic meta count (u32) + u32s [action|flags|insert_lo|insert_hi]; novelty count (u32)
+ (code,count) pairs; params count (u32) + online, target, adam_m, adam_v floats;
mask byte count (u32) + bytes; utility count (u32) + floats; FNV-1a64 checksum (u64)
over ALL preceding bytes. Little-endian, field-by-field (no struct padding).
- Save: tmp file + fsync + rename (primary→backup, tmp→primary), then validate by
  re-reading; on failure restore backup. Load: primary → backup → Missing.
  Corrupt/legacy files renamed `.bad` / `.legacy-v1` (evidence preserved).
- `validateFile` returns Ok/Corrupt/Incompatible (topology vs config); version field 1
  or 2 both accepted with v2 layout (migration path); magic "SIRCPT01" = legacy v1 →
  cannot map (pre-recurrent layout) → preserved, fresh organism, loud log.
- Save flow in app: `agent.exportState(state)` then fill app-owned fields
  (homeo snapshot, cheese_total, maze_generations, checkpoints_saved, timestamp,
  active_runtime_ms = persisted + session elapsed, last_consolidation_step,
  consolidation_cycles) then `store.save(state)`. Load: `store.load(&state)` → if Ok:
  `agent.importState(state)`, `sim.setLifetimeSteps/CheeseTotal/MazeGenerations`,
  `sim.homeostasis().restore(state.homeo)`, `metrics.setWallHitsTotal` (count is
  transient; keep), RNG: `rng.restoreState(state.rng_state)` **BEFORE constructing
  sim/agent** (app creates sim/agent via unique_ptr AFTER load — do NOT construct them
  as members in the constructor order that draws RNG first).
- IMPORTANT app ordering: cfg → logger → rng(seed) → load checkpoint → restore rng
  state → THEN create sim + agent → import state into agent → continue.

## 8. RENDERING / CONTROLS (renderer.{h,cpp} done; app must wire)

- Palette: bg 0x141414, wall 0x3a3a3a, floor 0x565656, rat 0xd8d8d8, rat-rest
  0x9a9a9a, nose 0x141414, cheese 0xb0b0b0 (notch 0x8c8c8c), text 0x99/0xcc.
- Maze at (8, 36), tile from config; panel at maze_x + maze_w*tile + 16.
- 3×5 bitmap font (utility/bitmap_font.{h,cpp}), scale 2 (char cell 8×10, line 14).
- Renderer::render(sim, panel); PanelData filled by app each frame (see renderer.h).
- Controls: Space pause, R new maze (no weight reset), S checkpoint now, D debug
  panel, Esc / window close = save + exit. Window title "Self Improving Rat".
- Headless (SIR_HEADLESS=1): skip SDL entirely. Screenshot mode: run normally with
  SDL_VIDEODRIVER=dummy + SIR_SCREENSHOT path; after ~30 frames save BMP and continue
  (renderer.saveScreenshot).

## 9. TESTS TO WRITE (tests/, tiny framework in tests/test_framework.h)

Base: maze validity/connectivity (BFS), collision handling, observation vector
(shape/values/bounds), NN forward (determinism, known-zero input), gradient sanity
(**GRU finite-difference gradient check — critical**, also for NeuralNet::backward
incl. pred sigmoid chain), replay buffer bounds/eviction, serialization roundtrip,
corrupt/truncated/NaN/checksum rejection, backup fallback, determinism (same seed →
identical trajectory + identical checkpoint bytes), config parsing/clamping.
Extension: homeostasis bounds + deterministic updates, cheese restores energy/hunger,
stress after collision, recurrent-state dims, episodic capacity + replacement,
curiosity reward bounds, novelty decay (repeat exposure → lower), prediction output
dims + loss stability, developmental schedule continuity (never 0, monotone), age
persistence, topology + mask serialization, plasticity hard limits + safe prune/rewire
+ rollback on instability, consolidation op limits, checkpoint backward-compat
(version=1 with v2 layout loads; incompatible topology rejected), full AL state
restore, no unbounded growth (replay/episodic/novelty stay at cap over many steps).
`./run.sh --test` builds+runs sir_tests. Tests must NOT need SDL.

## 10. NEXT SESSION — ACTION LIST (in order)

1. **Update `config/default.cfg`** with all organism keys (mirror Config defaults;
   group with comments; keep existing keys).
2. **Write `src/app/application.{h,cpp}`** per §3/§7/§8 (main loop: fixed-timestep
   accumulator, sim_steps_per_second pacing, render at fps with SDL_Delay sleep —
   no busy wait; max_catchup_steps_per_frame cap; autosave every
   autosave_interval_seconds; pause; signals via `sir::g_signal_count` atomic — declare
   in application.h or utility, increment in main.cpp handlers, poll in loop; second
   signal → immediate `_Exit(130)`; exception boundary in main; final checkpoint save
   on exit; screenshot + max-steps + headless hooks; PanelData fill; consolidation
   controller per §3; keep a ring of recent observations (last 64) for
   plasticityEvaluate; track adaptation_steps after regen).
   Add `void noteConsolidationCycle()` to Agent (or manage the counter in the app and
   write it into AgentState at save — prefer this, no agent change needed).
   Scale `grad_p` by `cfg_.prediction_loss_weight` in Agent::trainBatchOn (small edit).
3. **Write `src/main.cpp`**: parse `--config PATH`; env hooks SIR_SEED/SIR_HEADLESS/
   SIR_MAX_STEPS/SIR_SCREENSHOT/SIR_CONFIG; install SIGINT/SIGTERM handlers;
   try/catch around Application::run with cleanup + checkpoint attempt.
4. **Write `CMakeLists.txt`**: C++20, -Wall -Wextra -Wpedantic, Release default;
   static lib `sir_core` (utility+organism+simulation+learning+persistence), exe `sir`
   (+rendering+app+main+SDL2), exe `sir_tests` (tests+core only, no SDL);
   SDL2 resolution order: pkg-config → SIR_SDL2_PREFIX (include `${PREFIX}/usr/include`,
   lib `${PREFIX}/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}` fallback
   x86_64-linux-gnu/aarch64-linux-gnu) → clear error. `-O2`; `-fsanitize` optional.
   Note: `nn_hidden` config key exists but is unused legacy (kept for compatibility).
5. **Write `run.sh`**: see §1 (detect cmake/g++/pkg-config/sdl2; `--install-deps` runs
   apt-get only when requested; `--test` builds+runs sir_tests; `--clean` removes build/
   only; corrupted build dir → rm -rf build + reconfigure once; exec the binary so
   signals/exit codes propagate; `set -u`; cd to script dir; pass-through args; keep
   SIR_* env vars inherited). Must NOT touch data/checkpoints.
6. **Write tests** (list in §9). Then build → fix compile errors → run tests → headless
   smoke + screenshot (see §2) → fix runtime issues (watch for: mask init, RNG order,
   checkpoint roundtrip, GRU gradient test failures).
7. **Write README.md** (spec §17): purpose, Ubuntu requirements, run instructions,
   controls, architecture, simulation rules, exact observation vector, action space,
   exact reward function (formula §3), network architecture + learning algorithm,
   checkpoint format, recovery behavior, resource limits, config table, known
   limitations, how improvement is measured, honest note that long-term improvement is
   experimental and not guaranteed, artificial-life framing (no consciousness claims).
8. **Update `.agent_memory/GENERAL_PROGRESS.md`**, commit everything (first commit:
   `git add -A && git commit` — .gitignore excludes build/, .deps/, data/checkpoints/*,
   data/logs/*).

## 11. GOTCHAS / KNOWN ISSUES TO WATCH

- `default.cfg` outdated (see action 1).
- Nothing compiled yet — compile errors expected in newest files.
- `Agent::trainBatchOn` pred gradient weight not yet applied (see action 2).
- Renderer: `formatInt` uses std::to_string (fine); PanelData.checkpoints_saved +
  episode_count added to renderer.h — app must fill them.
- `Simulation::pushFrame` pushes 2–3 frames on cheese/regen steps (intended: next obs
  reflects the new cheese/maze; transition done=true so no learning distortion).
- `last_visit_step_` uses pre-increment numbering (`lifetime_steps_+1` stored before
  the increment; checks use the same convention) — do not "fix" without re-verifying
  the determinism test.
- Epsilon/Adam/lr use age scales from Development (continuous, never 0).
- `AgentState::allFinite()` is inline in agent_state.h (needs <cmath>).
- Checkpoint loader requires `simulation/observation.h` for kObservationBase.
- Do NOT add pathfinding to the agent; BFS exists only in Maze (validation/tests).
- `.deps/` is build aid; never commit; never delete data/checkpoints by hand.

## 12. SAFETY

- No root operations; no system changes beyond (optionally) apt for the user's own
  `--install-deps` run. No network. Single-threaded app. All buffers fixed-size.
- Data: checkpoints in data/checkpoints are app-managed; corrupt ones renamed .bad/
  .legacy-v1, never deleted silently; primary/backup invariant must be preserved in
  future edits (atomic write + validate-after-write + backup restore).
