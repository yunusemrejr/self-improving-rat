#pragma once

#include "utility/config.h"
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace sir {

struct LearningNote {
  std::string title;
  std::string category;
  std::string explanation;
  std::string latex;
  std::string notation;
  std::string source; // Maintainer provenance; not displayed in the habitat.
};

size_t learningNoteCount();
LearningNote learningNote(size_t index, const Config& cfg, double epsilon);
uint64_t noteReadingTime(const LearningNote& note);
std::vector<std::string> wrapNoteText(const std::string& text, size_t columns);

// Presentation-only RNG and clock: never consume the organism's RNG or steps.
// A shuffle bag shows every note before recycling, without an adjacent repeat.
class LearningNoteDeck {
 public:
  explicit LearningNoteDeck(uint32_t seed);
  size_t current() const { return order_[cursor_]; }
  bool held() const { return held_; }
  void start(uint64_t now, uint64_t duration);
  bool due(uint64_t now) const;
  uint64_t remaining(uint64_t now) const;
  void toggleHold(uint64_t now);
  void next();

 private:
  std::mt19937 rng_;
  std::vector<size_t> order_;
  size_t cursor_ = 0;
  bool held_ = false;
  uint64_t deadline_ = 0;
  uint64_t held_remaining_ = 0;
};

}  // namespace sir
