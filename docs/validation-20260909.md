# September 9, 2026 validation

This update preserves the existing 30-input, GRU-16 topology and improves
learning correctness, exploration, bounded replay, retention and rendering.
It uses portable C++20 and SDL2 software rendering, with no GPU or ML runtime.

## Matched online behavior

100,000 simulation steps per seed, fresh checkpoints, identical 13×9 maze
settings and reward-cheese value. The baseline is the actual working tree
at the start of this update, including the earlier Huber/BPTT work. Both
binaries were built in Release mode with GCC 15.2. Source and config for the
baseline were captured before editing. The final harness freezes its binary
and config at launch so another build cannot silently mix implementations.

| Seed | Before | Updated |
| ---: | ---: | ---: |
| 1 | 23 | 264 |
| 2 | 57 | 188 |
| 3 | 27 | 191 |
| 42 | 69 | 323 |
| 12345 | 10 | 201 |
| **Total** | **186** | **1,167** |
| **Mean** | **37.2** | **233.4** |

**6.27× as many cheeses at the same interaction budget.** This is a whole-system
behavior comparison, including wall masks, UCB-style episodic exploration,
reward shaping, retention and neural learning. It is not evidence that neural
learning alone improved 6.27× or that any component alone caused the gain.
Only five training seeds were used; no significance or universal superiority
claim is made. The improved default still needs exploration to avoid stalls.

Raw data: [before](benchmarks/before.csv), [after](benchmarks/after.csv).

## Frozen weights on unseen mazes

Each of the five final trained models was evaluated for 5,000 steps on maze
seeds 1001–1010. The evaluator never updates parameters, Adam moments, replay
priorities or training counters, and asserts that parameters/moments/counters
remain unchanged. Memory-assisted behavior retains its transient recurrent
state and episodic action counts; epsilon is fixed at 0.05. Greedy evaluation
disables both epsilon exploration and the count bonus. The untrained control
uses the same memory mechanism and epsilon as the trained controller.

| Controller | Mean cheese per 5,000 steps | Total over 50 runs |
| --- | ---: | ---: |
| Legal random walk | 8.70 | 435 |
| Untrained network + episodic exploration | 0.20 | 10 |
| Trained network + episodic exploration | 3.78 | 189 |
| Trained greedy network | 0.80 | 40 |

The trained controller improves over the untrained network with the same
exploration settings, but **does not beat a legal random walk on these unseen
mazes**. Its pure greedy policy generalizes poorly. Thus the online improvement
is substantial, while reliable frozen-policy navigation remains unsolved.
The ten environment seeds are reused for each of five model seeds; they are
not fifty independent environment seeds. The random and untrained controls
are repeated for pairing. [Full results](benchmarks/frozen.csv).

## Edge resource evidence

The default network contains 2,596 float32 parameters, about 10.1 KiB per
network. The online/target pair, Adam state, replay and scratch storage remain
bounded. No quantized inference, microcontroller, battery, thermal or mobile
accelerator claims are made.

The five final 100k-step runs took 15.48–16.24 seconds wall time and
15.02–15.75 seconds user CPU, with 18,064–18,680 KiB peak RSS on this host.
There were 56,058 neural updates per run. These are local process measurements,
not performance guarantees for other hardware. More recurrent training costs
more CPU than the earlier baseline; the main gain is cheese per interaction,
not a claimed speedup. Replay sampling now scales as O(batch × log capacity),
with no per-sample cumulative-distribution allocation.

## Persistence and correctness

- The default 4,096 replay transitions, priorities and sequence boundaries
  survive checkpoint export/import. Larger memories save their newest entries
  within an 8 MiB replay budget. Reduced capacity restores the newest entries.
- v3 files include checksummed replay. Existing v2 organism files migrate
  without changing topology or dropping weights, Adam state or episodic memory.
- A **copy of the user's existing checkpoint** successfully restored its
  44,403-step organism, ran one further step and saved in the new format. The
  original live checkpoint files were not changed during verification.
- Bad saves are validated before promotion and preserve both existing valid
  generations. Unsupported formats cannot be overwritten by a fresh organism.
  File and directory fsync protect checkpoint promotion.
- Tests cover replay wrap/probabilities, transactional invalid imports,
  chronological observation stacks, correct recurrent bootstrap, masked next
  actions, burn-in/sequence terminal boundaries, useful-memory retention,
  utility EMA persistence, dormant-gradient masking and scent reward loops.
- The Release CTest suite passes: 88 tests, 4,713,025 checks, zero failures. AddressSanitizer/UBSan checks cover checkpoint
  handling, replay restore, recurrent TD, sequence boundaries and utility masks.
  LeakSanitizer is disabled because this runner reports it cannot work under
  ptrace; leak detection is not claimed.

## Visual verification

Inspected actual software-rendered frames at 1060×680, 640×480 and 320×240,
including four facing directions, pause, rest and diagnostics states. The
rat has grey fur, pink ears and tail, a pointed muzzle, eyes and whiskers;
cheese uses golden triangular faces and shaded holes. Rounded connected
corridors replace the square monochrome grid. The normal panel exposes real
memory, replay, neural-update and save counts; D shows training diagnostics.
At very small windows the maze fits with smaller cells and compact counters.

### Learning reader follow-up

The subsequent reader update adds 26 shuffled explanations with source
provenance, configuration-aware examples, and typeset formulas with symbol
guides. Very high or low probabilities use bounds instead of rounding to
certainty or impossibility. Possible emergent routes and generalization are
described without inventing a probability of future mastery.

The Release suite now contains 93 core tests, plus an SDL integration target.
Tests check mathematical glyphs and layout bounds, probability examples and
disabled settings, no-repeat shuffle bags, hold/resume timing, keyboard repeat,
mouse controls and unchanged simulation RNG. The SDL test renders every note
at default and compact sizes, including while the simulation is paused.
The actual application also saved and resumed an isolated temporary checkpoint.

The default window is now 1060×960. Compact requested sizes expand to at least
640 pixels wide and enough height for the longest note, avoiding tiny type or
clipped explanations. Formula and notation are placed beside each other when
space allows, and the habitat stays fixed as notes rotate. Screenshots of the
software renderer were inspected for readable formulas and unobstructed
controls. These presentation changes do not change the learning benchmark above.

## Reproduce

```bash
./run.sh --test
TAG=verify KEEP_RUNS=/tmp/rat-evaluation tools/bench.sh 100000 1 2 3 42 12345
# Use a retained config path printed by the previous command:
./build-bench/sir_eval /tmp/rat-evaluation/seed-1-XXXXXX/config.cfg 5000 1001 10
```

Research grounding: [prioritized experience replay](https://arxiv.org/abs/1511.05952)
and [recurrent replay / burn-in](https://willdabney.com/publication/r2d2/).
This is a small local adaptation of established techniques, not a reproduction
of a distributed state-of-the-art agent.
