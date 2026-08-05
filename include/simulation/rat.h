#pragma once
// Rat state and the 4-action movement space. The rat is a physical agent in
// the maze: it can move one cell per step in a cardinal direction, or fail to
// move when the target cell is a wall.

#include <cstdint>

namespace sir {

enum class Action : uint8_t { Up = 0, Down = 1, Left = 2, Right = 3, Count = 4 };

struct Position {
  int x = 0;
  int y = 0;
  bool operator==(const Position&) const = default;
};

class Rat {
 public:
  void setPosition(int x, int y) {
    pos_.x = x;
    pos_.y = y;
  }
  const Position& position() const { return pos_; }

  void setFacing(Action a) { facing_ = a; }
  Action facing() const { return facing_; }

  void setLastAction(Action a) { last_action_ = a; }
  Action lastAction() const { return last_action_; }

  void setWallHitLastAction(bool v) { wall_hit_ = v; }
  bool wallHitLastAction() const { return wall_hit_; }

  // Walking animation frame, toggled on every successful move.
  void advanceWalkFrame() { walk_frame_ = (walk_frame_ + 1) % 2; }
  int walkFrame() const { return walk_frame_; }

 private:
  Position pos_{0, 0};
  Action facing_ = Action::Up;
  Action last_action_ = Action::Up;
  bool wall_hit_ = false;
  int walk_frame_ = 0;
};

}  // namespace sir
