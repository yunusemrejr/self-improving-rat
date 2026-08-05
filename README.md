# Self Improving Rat

An Ubuntu-native C++20/CMake/SDL2 desktop application: a persistent
artificial-life rat (reinforcement learning + homeostasis + world model +
recurrent memory + episodic memory + structural plasticity + consolidation)
navigating procedurally generated mazes, finding cheese, and persisting its
developmental state across sessions.

The rat is a **learning agent, not a scripted demo**: it observes only local
signals, chooses actions from learned Q-values, and improves through genuine
online weight updates. The project's claim is **persistent adaptation
attempts with bounded resources — not monotonic intelligence growth**.
Long-term improvement is experimental and not guaranteed.

---

## Requirements and setup

- Ubuntu (tested on 26.04), `cmake >= 3.16`, a C++20 compiler (`g++ >= 12`),
  `make` or `ninja`, and SDL2 development files (`libsdl2-dev`). The SDL2
  runtime library is usually already installed.
- If `libsdl2-dev` is missing:
  - `./run.sh --install-deps` (asks for sudo — opt-in only), or
  - point the build at a local dev extraction:
    `SIR_SDL2_PREFIX=/path/to/extracted ./run.sh`

## Build and run

```bash
./run.sh                 # build (if needed) and run the application
./run.sh --test          # build and run the test suite (no SDL needed)
./run.sh --clean         # remove only the build directory
./run.sh --sanitize      # build and run with AddressSanitizer + UBSan
```

`run.sh` never touches `data/checkpoints` or `data/logs`. A damaged build
directory is repaired automatically (removed and reconfigured once).

Environment hooks (inherited by the application):

| Variable | Meaning |
|---|---|
| `SIR_SEED=N` | non-zero = deterministic run (maze, cheese, exploration) |
| `SIR_HEADLESS=1` | run without any window (diagnostic mode) |
| `SIR_MAX_STEPS=N` | clean exit after N simulation steps |
| `SIR_SCREENSHOT=path.bmp` | save one rendered frame after ~30 frames, then continue |
| `SIR_CONFIG=path` | alternate config file |
| `SIR_SDL2_PREFIX=path` | local SDL2 dev extraction for the build |

## Controls

| Key | Action |
|---|---|
| `Space` | pause / resume (no state is lost; timing stays fixed-step) |
| `R` | generate a new maze (learned weights are untouched) |
| `S` | save a checkpoint immediately |
| `D` | toggle the diagnostics panel (all values are real measurements) |
| `Esc` / close window | save a checkpoint, then exit |

`SIGINT` and `SIGTERM` also trigger the same graceful save-and-exit; a second
signal forces immediate exit.

---

## What the rat is

The rat lives in a maze with one piece of cheese. Every simulation step it
chooses one of four actions (Up, Down, Left, Right). It **cannot see the maze
map, cheese coordinates, or any precomputed path** — it receives a 28-channel
local observation (below) and its own internal state. Movement into a wall
is blocked (collision). Collecting cheese yields the largest reward; the
maze regenerates every N cheeses (default 6) without touching learned
weights.

The rat has a simulated body with bounded homeostatic variables that
influence both the observation and the reward, so it must learn to balance
cheese-seeking with its own needs:

| Variable | Bounds | Effect |
|---|---|---|
| energy | [0,1] | decays over time and with movement; cheese restores |
| hunger | [0,1] | grows over time; cheese reduces |
| fatigue | [0,1] | grows with movement; recovers while resting (consolidation) |
| stress | [0,1] | rises on wall collisions (worse when repeated); falls on success/cheese |
| curiosity need | [0,1] | grows over time; consumed by novel experience |
| satisfaction | [0,1] | rises on cheese; reverts toward 0.5 |
| prediction uncertainty | [0,1] | EMA of recent world-model prediction error |

All variables are clamped to [0,1] every step and are restored from
checkpoints.

### Exact observation vector

One base frame = 30 channels, all in [0,1]; the network input is
`30 × observation_frames` (shipped config uses 2 frames = 60 inputs):

```
[0..7]   wall bits: N, NE, E, SE, S, SW, W, NW (1 = blocked)
[8..11]  cheese scent N, E, S, W: max over cheeses of
         (1 - dist/(radius+1)) * max(0, dot(dir, offset)/manhattan)
         (distance-only; smell travels around corners, walls do not block it)
[12..15] last action one-hot (Up, Down, Left, Right)
[16]     wall-hit flag of the last action
[17]     revisit signal (1 - steps_since_last_entry/8)
[18]     last navigation reward mapped to [0,1] via (clamp(r,-1,1)+1)/2
[19..25] homeostasis: energy, hunger, fatigue, stress, curiosity need,
         satisfaction, prediction uncertainty
[26]     recent novelty signal
[27]     resting flag (1 during consolidation)
[28..29] proprioception: own x/y position normalized to [0,1] (a rat's
         kinesthetic sense of place; not hidden maze information)
```

No channel encodes the maze map, cheese coordinates, path values, or any
other privileged information (verified by tests that mutate far-away maze
state and assert the observation is unchanged). The two proprioception
channels carry the rat's own body position so the task is a well-posed
observable MDP; since the maze regenerates periodically, they cannot encode
a fixed layout.

### Exact reward (per step, clamped to [-12, 12])

```
r_total = r_nav                                   # navigation
        + r_homeo                                 # body needs
        + r_curiosity                             # intrinsic
        + r_prediction                            # world-model bonus
        - collapse_penalty                        # anti-lockup (contextual)

r_nav      = +10.0 on cheese | -0.2 on wall hit | -0.02 per step
             | -0.05 when re-entering a cell within 4 steps
r_homeo    = -0.002*(1-energy) - 0.004*hunger - 0.002*fatigue
             - 0.004*stress^2 + 0.001*(satisfaction - 0.5)
r_curiosity = 0.02 * novelty * (0.5 + 0.5 * curiosity_need)   # capped small
r_prediction = 0.01 * (1 - uncertainty)
collapse_penalty = 0.01 when action entropy over the last 2000 steps < 0.2
             AND recent average reward < -0.05 AND lifetime steps > 1000
```

The curiosity bonus is deliberately far smaller than the cheese reward, so
exploration can never dominate survival objectives.

---

## Learning system

Architecture (all single-threaded, deterministic given a seed):

```
observation (28*frames)
      |
   [ GRU: 16 hidden ]  -> working memory h_t = GRU(x_t, h_{t-1})
      |___________________-> policy head: 4 Q-values  (4 floats)
      |___________________-> prediction head: 16 sigmoid outputs
```

- **Policy**: double-DQN (action chosen by the online net on the next state,
  value taken from a slowly-updated target net). Adam optimizer
  (β1=0.9, β2=0.999, ε=1e-8, bias-corrected), global gradient clipping at
  norm 1.0, soft target updates (τ=0.02). Training runs every 4 steps with a
  batch of 32 transitions sampled from a fixed-capacity replay buffer (8192).
- **Recurrent memory**: each replay transition stores the recurrent state
  used when it was collected (truncated BPTT-1 with stale states — a
  documented approximation). The recurrent state is reset only at session
  start, after maze regeneration, and after consolidation (lifecycle
  boundaries). Recurrent parameters are trained and persisted.
- **World model / prediction head** (16 outputs): predicts the next
  observation's wall bits and scent (12 channels), the homeostasis deltas
  (3), and the external reward mapped to [0,1] (1). Predictions are made
  *before* the next state is observed; the targets come from the real
  transition; the prediction gradient is scaled by `prediction_loss_weight`
  (0.3) so it never dominates the policy objective.
- **Curiosity**: novelty = 0.5 × state novelty + 0.5 × prediction error.
  State novelty comes from a bounded quantized-state hash table
  (`novelty = 1/sqrt(visit_count)`; capacity 1024 with count-halving), so
  repeated exposure loses novelty. Prediction error is the normalized MSE of
  the world model, bounded to [0,1]. Curiosity reward = 0.02 × novelty
  (modulated by the curiosity need), capped and small.
- **Episodic memory**: fixed capacity 256. Significant transitions are stored
  (cheese, |reward|>1, novelty>0.6, prediction error>0.3); when full, the
  least significant entry is replaced (ties: oldest). Entries are compact
  (observation, next observation, recurrent state, action, reward, flags).
- **Structural plasticity** (only during consolidation): connections carry
  bit masks (active/dormant) and utility traces (EMA of |gradient ×
  weight|). Pruning removes demonstrably weak active connections (capped per
  consolidation, never below a hard active-connection cap), reactivation
  restores previously useful dormant connections with small fresh weights,
  and useful connections are strengthened mildly. Every change is evaluated
  with bounded forward passes and rolled back on instability. Topology is
  serialized in checkpoints and restored on load.
- **Consolidation**: triggered after 4 cheeses or when fatigue ≥ 0.9, with a
  minimum interval of 2000 steps. During consolidation the rat rests
  (fatigue recovers), bounded training batches (max 60 ops over max 120
  ticks, spread out so the GUI stays responsive) mix episodic memory with
  replay, structural plasticity runs, and a checkpoint is saved.

### Exploration

Epsilon-greedy: 0.9 at birth, linearly decaying to 0.05 over 50,000 steps,
multiplied by an age-based scale (never below the floor). Both exploration
and the learning rate follow smooth age schedules — there are no discrete
life stages, and learning never switches off.

### Development

Age = lifetime steps (persisted). Learning rate, exploration, and plasticity
rates follow continuous, monotone functions of normalized age
`tau = min(1, steps/200000)` that never reach zero, so adaptation remains
possible throughout life.

---

## Persistence

- Binary checkpoint format `SIRCPT02` (little-endian, field-by-field, no
  struct padding) storing: topology, hyperparameters, lifetime counters,
  homeostasis snapshot, age/runtime, consolidation history, structural
  plasticity history, online + target network parameters, Adam moments,
  connection masks, utility traces, episodic memory, novelty table, and the
  full RNG state.
- **Atomic writes**: write to `.tmp`, `fsync`, rename (previous valid
  checkpoint is promoted to `.bak` first), then re-read and validate the new
  file; on failure the backup is restored. The last valid checkpoint is
  never overwritten by partial data.
- **Checksums**: FNV-1a64 over the whole file; size caps and per-vector
  length validation on load.
- **Recovery order**: primary → backup → fresh organism. Corrupt files are
  renamed `.bad` (legacy pre-recurrent files `.legacy-v1`) — evidence is
  preserved, never silently deleted. Checkpoints with an incompatible
  topology are left in place (never destroyed) and a fresh organism starts.
- Checkpoints are written on autosave (default every 15 s), on `S`, on
  consolidation, and on every clean exit (window close, Esc, SIGINT,
  SIGTERM, `SIR_MAX_STEPS`).

## Resource limits (verified)

| Resource | Limit |
|---|---|
| replay buffer | 8192 transitions (preallocated ring, ~2.6 MB) |
| episodic memory | 256 entries |
| novelty table | 1024 slots |
| log file | 512 KiB then rotate (keeps 2 files) |
| checkpoint files | primary + one backup, ~230 KiB each |
| simulation rate | 12 steps/second (capped; `max_catchup_steps_per_frame` bounds catch-up) |
| render rate | 30 frames/second (capped, `SDL_Delay` — no busy-wait) |
| training | 1 batch per 4 steps; consolidation ops capped |
| threads | 1 |

Measured on a 500,000-step headless run: resident memory flat at ~2.3 MB,
1 thread, 3 file descriptors, log 3.4 KiB.

## Tests

`./run.sh --test` builds and runs the suite (47 tests, no SDL needed):

- maze validity/connectivity across many seeds, collisions, cheese
  reachability, observation bounds and **no-cheating isolation**
- deterministic trajectories for fixed seeds
- homeostasis bounds and event effects, development schedules, novelty decay
- GRU and full-network **finite-difference gradient checks**
- genuine learning: parameters change after training, stay frozen when
  paused, and keep changing after checkpoint restore; prediction gradients
  update real parameters
- replay buffer wraparound, episodic memory capacity/replacement
- plasticity hard limits, consolidation bounds
- checkpoint round-trip, corruption, checksum, NaN rejection, backup
  fallback, incompatible topology, atomicity, deterministic bytes
- long-run boundedness (no unbounded growth)

The suite runs clean under AddressSanitizer + UndefinedBehaviorSanitizer.

## Configuration

`config/default.cfg` documents every key with safe built-in defaults (a
partial or missing file is fine — invalid keys/values are logged and
ignored). Notable groups: `[simulation]` maze/cheese/revisit; navigation
rewards; learning hyperparameters; homeostasis rates; reward composition;
prediction/curiosity; episodic memory; lifelong development; consolidation;
structural plasticity; persistence/logging.

## Known limitations

- **No guaranteed improvement.** The system guarantees persistent,
  resource-bounded adaptation attempts; it does not guarantee monotonic
  intelligence growth. Whether the rat reliably improves at finding cheese
  depends on the maze difficulty and must be measured per run.
- The world model predicts a coarse, compressed target (16 channels); it is
  a novelty source, not a complete environment model.
- The recurrent state uses a stale-state approximation (truncated BPTT-1).
- `nn_hidden` is a legacy config key kept for compatibility; it is unused.
- The rat has no long-term spatial map: cheese scent is distance-only and
  does not propagate through walls, and there is no place memory beyond the
  recent-revisit signal.
- This is an artificial-life simulation. The rat has no consciousness, and
  no consciousness-like behavior is claimed.
