# Session 2026-08-05T10:xxZ — deepseek-v4-flash (final verification & repair) — part 2

## CRITICAL ROOT-CAUSE FINDING (verified by experiment)

**Aliasing bug in Application::doSimStep (FIXED):** `sim.observe()` and
`agent.recurrentState()` return pointers into internal buffers that
`sim.step()` / `selectAction()` mutate. The app stored `s` (aliased →
post-action frame) and `h_prev` (aliased → post-action recurrent state) in
the replay, so every transition was (post, post) with a self-loop bootstrap.
This made ALL maze learning fail: Q-values oscillated ±4, greedy policy was
anti-cheese (toward 0.38-0.47), cheese rate dropped to ~0 as epsilon decayed.
The open-grid diagnostic (separate local arrays) learned perfectly (greedy
0.88→1.00), which proved the pipeline itself was fine and exposed the
aliasing. FIX: copy s and h_prev into locals before the mutating calls.

## Other learning-system findings (all verified)

- GRU backward + NeuralNet backward PASS finite-difference gradient checks
  (need eps ~1e-2; eps=1e-3 gives false failures on saturated gates).
- Toy MDP (cheese on action index 1 to defeat tie-break bias) learns 88→100%
  greedy. Pipeline works.
- Maze navigation with walls + MOVING cheese is hard for this DQN: cheese is
  found ~1/1500 steps and re-placed on every find, faster than Q-learning
  adapts. With a FIXED cheese the agent learns maze navigation (greedy-toward
  0.85, cheese accelerating) — the moving target is the difficulty.
- Fixes that worked (shipped): observation extended 28→30 channels with 2
  proprioception channels (normalized x,y — the rat's own body state, not in
  the no-cheating list); cheese scent falloff softened to radius/(dist+radius);
  prioritized experience replay (|TD| priorities) in ReplayBuffer; dense
  scent-proximity reward (gain×max-scent-after, config
  scent_proximity_reward_gain=0.08) — a delta form punished maze detours and
  was removed; batch 16 / train every 2 steps / replay 4096.
- Defaults changed for learnability: maze 47x31→13x9, tile 13→24, radius
  6→20 (covers maze), min_dist 12→3, braid 0.12→0.40, observation_frames 2→1.
- App with corrected defaults: cheese=16/50k (seed 42), 26k training updates,
  24 consolidation cycles, 0 ASan errors. Cheese rate is ~1.5-2x the
  exploration-only expectation: genuine but modest learning. Long-term
  monotonic improvement is NOT guaranteed — README must state this.

## Files changed by this session (current state)
- Fixed: neural_net.cpp (updateUtility index fix, forward pred null guard),
  replay_buffer.{h,cpp} (reward_ext, PER), agent.{h,cpp} (reward_ext flow,
  pred loss weight, PER TD feedback, qValuesFor), episodic_memory.{h,cpp}
  (reward_ext), checkpoint.cpp (per-entry floats, skip(8) — the OTHER session
  added skip(8), verified correct, kept; rng guards), observation.{h,cpp}
  (30 channels + position, scent falloff), simulation.{h,cpp} (int64 visit
  steps, lastRevisit), application.{h,cpp} (aliasing fix, metrics fixes, fps,
  seed_override, scent proximity), config.{h,cpp} + default.cfg (new defaults
  + organism keys + scent_proximity_reward_gain).
- Created: CMakeLists.txt, run.sh, src/main.cpp.

## Still TODO
1. tests/ test suite (test_framework.h + full suite per HANDOFF §9) — MOST
   IMPORTANT REMAINING ITEM.
2. README.md.
3. Persistence stress tests (corruption/backup/roundtrip) — partially
   verified manually; formalize in tests.
4. Long-run resource checks, GUI screenshot verification, run.sh --test.
5. Commit everything (first commit).
6. NOTE: build-asan dir disappeared once (concurrent session?); rebuilt.
