#include "simulation/observation.h"

#include <algorithm>
#include <cmath>

namespace sir {

namespace {

// Unit direction vectors for the four cardinal directions.
constexpr int kDirX[4] = {0, 0, -1, 1};
constexpr int kDirY[4] = {-1, 1, 0, 0};

}  // namespace

void buildObservation(const Maze& maze, int rat_x, int rat_y,
                      Action last_action, bool wall_hit, float revisit_signal,
                      float last_reward_clipped,
                      const std::vector<Position>& cheeses, int sensory_radius,
                      const ObservationContext& ctx, float* out) {
  // [0..7] wall bits around the rat (8-neighborhood).
  const int nbr_x[8] = {0, 1, 1, 1, 0, -1, -1, -1};
  const int nbr_y[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
  for (int i = 0; i < 8; ++i) {
    out[i] = maze.isWall(rat_x + nbr_x[i], rat_y + nbr_y[i]) ? 1.0f : 0.0f;
  }

  // [8..11] cheese scent channels. For each cheese within the sensory radius
  // (manhattan distance), the direction-weighted closeness
  //   strength = (radius / (dist + radius)) * max(0, dot(dir, offset)/dist)
  // is accumulated (max over cheeses) per cardinal direction. The hyperbolic
  // falloff keeps the gradient informative across the whole radius (a linear
  // falloff is near zero at the edge); walls do not block smell.
  const float radius = static_cast<float>(std::max(1, sensory_radius));
  float scent[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (const Position& c : cheeses) {
    const float dx = static_cast<float>(c.x - rat_x);
    const float dy = static_cast<float>(c.y - rat_y);
    const float dist = std::fabs(dx) + std::fabs(dy);
    if (dist < 1.0f || dist > radius) continue;
    const float closeness = radius / (dist + radius);
    for (int d = 0; d < 4; ++d) {
      const float align = (dx * static_cast<float>(kDirX[d]) +
                           dy * static_cast<float>(kDirY[d])) /
                          dist;
      if (align > 0.0f) {
        scent[d] = std::max(scent[d], closeness * align);
      }
    }
  }
  for (int d = 0; d < 4; ++d) out[8 + d] = std::min(1.0f, std::max(0.0f, scent[d]));

  // [12..15] last action one-hot (Up, Down, Left, Right).
  for (int a = 0; a < 4; ++a) out[12 + a] = 0.0f;
  out[12 + static_cast<int>(last_action)] = 1.0f;

  // [16] wall-hit flag of the last action.
  out[16] = wall_hit ? 1.0f : 0.0f;

  // [17] revisit signal.
  out[17] = std::min(1.0f, std::max(0.0f, revisit_signal));

  // [18] last reward mapped to [0,1].
  out[18] = std::min(1.0f, std::max(0.0f, (last_reward_clipped + 1.0f) * 0.5f));

  // [19..27] organism internal state.
  out[19] = ctx.energy;
  out[20] = ctx.hunger;
  out[21] = ctx.fatigue;
  out[22] = ctx.stress;
  out[23] = ctx.curiosity_need;
  out[24] = ctx.satisfaction;
  out[25] = ctx.uncertainty;
  out[26] = std::min(1.0f, std::max(0.0f, ctx.novelty));
  out[27] = ctx.resting > 0.5f ? 1.0f : 0.0f;

  // [28..29] proprioception: the rat's own normalized position in the maze.
  out[28] = static_cast<float>(rat_x) / static_cast<float>(maze.width() - 1);
  out[29] = static_cast<float>(rat_y) / static_cast<float>(maze.height() - 1);
}

}  // namespace sir
