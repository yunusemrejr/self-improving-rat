#pragma once
// Observation vector construction. The rat never sees the maze map, cheese
// coordinates or any precomputed path; it receives only bounded local
// signals plus its own internal (homeostatic) state. See the .cpp file and
// README for the exact vector layout.

#include "simulation/maze.h"
#include "simulation/rat.h"

#include <cstddef>
#include <vector>

namespace sir {

// Base observation size (one frame). With `observation_frames` stacking, the
// network input is kObservationBase * frames.
//
// Layout of one base frame (30 channels, all in [0,1]):
//   [0..7]   wall bits: N, NE, E, SE, S, SW, W, NW       (1 = blocked)
//   [8..11]  cheese scent: Up, Down, Left, Right (action-aligned)  (0..1)
//   [12..15] last action one-hot: Up, Down, Left, Right
//   [16]     wall-hit flag of the last action
//   [17]     revisit signal (1 - steps_since_last_visit / 8)
//   [18]     last reward mapped to [0,1] via (clamp(r,-1,1)+1)/2
//   [19]     energy
//   [20]     hunger
//   [21]     fatigue
//   [22]     stress
//   [23]     curiosity need
//   [24]     satisfaction
//   [25]     prediction uncertainty
//   [26]     recent novelty signal (0..1)
//   [27]     resting flag (1 during consolidation)
//   [28]     x position normalized to [0,1] (proprioception)
//   [29]     y position normalized to [0,1] (proprioception)
//
// The two proprioception channels carry the rat's own body position (its
// kinesthetic sense of place), not any hidden maze information: the full
// maze map, cheese coordinates, path lengths and reachability remain
// inaccessible to the policy. They make the task a proper observable MDP so
// the recurrent policy can learn navigation (a real rat knows where it is).
inline constexpr int kObservationBase = 30;

// Per-frame channels that describe the organism's internal state.
struct ObservationContext {
  float energy = 0.8f;
  float hunger = 0.2f;
  float fatigue = 0.0f;
  float stress = 0.0f;
  float curiosity_need = 0.5f;
  float satisfaction = 0.5f;
  float uncertainty = 0.5f;
  float novelty = 0.0f;     // recent novelty signal
  float resting = 0.0f;     // 1 while consolidating
};

// Builds one base observation into `out` (kObservationBase floats, values in
// [0,1]). `cheeses` holds cheese positions; scent is distance-based and
// ignores walls (smell travels around corners), see observation.cpp.
void buildObservation(const Maze& maze, int rat_x, int rat_y,
                      Action last_action, bool wall_hit, float revisit_signal,
                      float last_reward_clipped,
                      const std::vector<Position>& cheeses, int sensory_radius,
                      const ObservationContext& ctx, float* out);

}  // namespace sir
