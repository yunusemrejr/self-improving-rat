#include "simulation/maze.h"

#include "simulation/rat.h"  // Position (used by reachability validation)

#include <stdexcept>
#include <utility>
#include <vector>

namespace sir {

namespace {
constexpr int kDirX[4] = {0, 0, -1, 1};
constexpr int kDirY[4] = {-1, 1, 0, 0};
}  // namespace

Maze::Maze(int width, int height) : w_(width), h_(height) {
  if (w_ < 5 || h_ < 5) {
    throw std::runtime_error("maze dimensions must be at least 5x5");
  }
  cells_.assign(static_cast<size_t>(w_) * h_, 1);
}

bool Maze::inBounds(int x, int y) const {
  return x >= 0 && y >= 0 && x < w_ && y < h_;
}

bool Maze::isWall(int x, int y) const {
  if (!inBounds(x, y)) return true;  // outside the maze behaves as a wall
  return cells_[static_cast<size_t>(y) * w_ + x] != 0;
}

bool Maze::isWalkable(int x, int y) const {
  return inBounds(x, y) && cells_[static_cast<size_t>(y) * w_ + x] == 0;
}

void Maze::setWallForTests(int x, int y, bool wall) {
  if (!inBounds(x, y)) return;
  cells_[static_cast<size_t>(y) * w_ + x] = wall ? 1 : 0;
}

void Maze::resetAllWalls() {
  for (auto& c : cells_) c = 1;
}

void Maze::generate(Rng& rng, double braid_probability) {
  if (w_ < 5 || h_ < 5) throw std::runtime_error("maze too small");
  resetAllWalls();

  // Iterative recursive backtracker on the odd grid.
  // Start at (1,1) (border cells stay walls).
  const int sx = 1, sy = 1;
  const int gx = (w_ - 1) / 2, gy = (h_ - 1) / 2;  // odd-grid extents (0-based)
  cells_[static_cast<size_t>(sy) * w_ + sx] = 0;

  struct Cell { int x, y; };
  std::vector<Cell> stack;
  stack.push_back({sx, sy});

  int dirs[4] = {0, 1, 2, 3};
  while (!stack.empty()) {
    Cell c = stack.back();
    int nx = 0, ny = 0;
    int chosen = -1;
    // Shuffle candidate directions.
    for (int i = 3; i > 0; --i) {
      int j = rng.uniformInt(0, i);
      std::swap(dirs[i], dirs[j]);
    }
    for (int d : dirs) {
      nx = c.x + kDirX[d] * 2;
      ny = c.y + kDirY[d] * 2;
      if (nx >= 1 && ny >= 1 && nx <= gx * 2 - 1 && ny <= gy * 2 - 1 &&
          cells_[static_cast<size_t>(ny) * w_ + nx] == 1) {
        chosen = d;
        break;
      }
    }
    if (chosen >= 0) {
      int wx = c.x + kDirX[chosen];
      int wy = c.y + kDirY[chosen];
      cells_[static_cast<size_t>(wy) * w_ + wx] = 0;
      nx = c.x + kDirX[chosen] * 2;
      ny = c.y + kDirY[chosen] * 2;
      cells_[static_cast<size_t>(ny) * w_ + nx] = 0;
      stack.push_back({nx, ny});
    } else {
      stack.pop_back();
    }
  }

  // Braiding: for each dead end, with probability p open a wall to an
  // adjacent corridor, creating loops. Removing walls keeps connectivity.
  if (braid_probability > 0.0) {
    for (int y = 1; y < h_ - 1; ++y) {
      for (int x = 1; x < w_ - 1; ++x) {
        size_t idx = static_cast<size_t>(y) * w_ + x;
        if (cells_[idx] != 0) continue;
        int open_neighbors = 0;
        for (int d = 0; d < 4; ++d) {
          if (isWalkable(x + kDirX[d], y + kDirY[d])) ++open_neighbors;
        }
        if (open_neighbors != 1) continue;  // not a dead end
        if (rng.uniform01() >= braid_probability) continue;
        // Pick a random wall between this dead end and another corridor.
        int candidates[4], n = 0;
        for (int d = 0; d < 4; ++d) {
          int nx = x + kDirX[d], ny = y + kDirY[d];
          if (inBounds(nx, ny) && !isWalkable(nx, ny) &&
              // The cell beyond must be walkable (connects two passages).
              isWalkable(x + kDirX[d] * 2, y + kDirY[d] * 2)) {
            candidates[n++] = d;
          }
        }
        if (n == 0) continue;
        int d = candidates[rng.uniformInt(0, n - 1)];
        cells_[static_cast<size_t>(y + kDirY[d]) * w_ + (x + kDirX[d])] = 0;
      }
    }
  }
}

int Maze::reachableCount(int sx, int sy) const {
  if (!isWalkable(sx, sy)) return 0;
  std::vector<uint8_t> seen(static_cast<size_t>(w_) * h_, 0);
  std::vector<Position> queue;
  queue.reserve(static_cast<size_t>(w_) * h_);
  seen[static_cast<size_t>(sy) * w_ + sx] = 1;
  queue.push_back({sx, sy});
  size_t head = 0;
  int count = 1;
  while (head < queue.size()) {
    Position p = queue[head++];
    for (int d = 0; d < 4; ++d) {
      int nx = p.x + kDirX[d], ny = p.y + kDirY[d];
      if (!isWalkable(nx, ny)) continue;
      size_t i = static_cast<size_t>(ny) * w_ + nx;
      if (seen[i]) continue;
      seen[i] = 1;
      ++count;
      queue.push_back({nx, ny});
    }
  }
  return count;
}

}  // namespace sir
