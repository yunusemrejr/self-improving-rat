#include "utility/bitmap_font.h"

#include <cstring>

namespace sir {

namespace {

// 3x5 glyphs, rows top to bottom, bit 0 = leftmost pixel.
// Layout: { character, 5 row bytes }.
const struct Glyph {
  char ch;
  uint8_t rows[5];
} kGlyphs[] = {
    {'A', {2, 5, 7, 5, 5}}, {'B', {6, 5, 6, 5, 6}}, {'C', {3, 4, 4, 4, 3}},
    {'D', {6, 5, 5, 5, 6}}, {'E', {7, 4, 6, 4, 7}}, {'F', {7, 4, 6, 4, 4}},
    {'G', {3, 4, 5, 5, 3}}, {'H', {5, 5, 7, 5, 5}}, {'I', {7, 2, 2, 2, 7}},
    {'J', {1, 1, 1, 5, 2}}, {'K', {5, 5, 6, 5, 5}}, {'L', {4, 4, 4, 4, 7}},
    {'M', {5, 7, 7, 5, 5}}, {'N', {5, 7, 7, 7, 5}}, {'O', {2, 5, 5, 5, 2}},
    {'P', {6, 5, 6, 4, 4}}, {'Q', {2, 5, 5, 6, 3}}, {'R', {6, 5, 6, 5, 5}},
    {'S', {3, 4, 2, 1, 6}}, {'T', {7, 2, 2, 2, 2}}, {'U', {5, 5, 5, 5, 7}},
    {'V', {5, 5, 5, 5, 2}}, {'W', {5, 5, 7, 7, 5}}, {'X', {5, 5, 2, 5, 5}},
    {'Y', {5, 5, 2, 2, 2}}, {'Z', {7, 1, 2, 4, 7}}, {'0', {2, 5, 5, 5, 2}},
    {'1', {2, 6, 2, 2, 7}}, {'2', {6, 1, 2, 4, 7}}, {'3', {6, 1, 6, 1, 6}},
    {'4', {5, 5, 7, 1, 1}}, {'5', {7, 4, 6, 1, 6}}, {'6', {3, 4, 6, 5, 2}},
    {'7', {7, 1, 2, 2, 2}}, {'8', {2, 5, 2, 5, 2}}, {'9', {2, 5, 3, 1, 6}},
    {' ', {0, 0, 0, 0, 0}}, {'.', {0, 0, 0, 0, 2}}, {':', {0, 2, 0, 2, 0}},
    {'-', {0, 0, 7, 0, 0}}, {'/', {1, 1, 2, 4, 4}}, {'%', {5, 1, 2, 4, 5}},
    {'#', {5, 7, 5, 7, 5}}, {'!', {2, 2, 2, 0, 2}}, {'(', {1, 2, 2, 2, 1}},
    {')', {4, 2, 2, 2, 4}}, {'_', {0, 0, 0, 0, 7}}, {'+', {0, 2, 7, 2, 0}},
    {'>', {4, 2, 1, 2, 4}}, {'<', {1, 2, 4, 2, 1}}, {'=', {0, 7, 0, 7, 0}},
    {'?', {6, 1, 2, 0, 2}},
};

// Fallback glyph (filled block) for unsupported characters.
const uint8_t kFallback[5] = {7, 7, 7, 7, 7};

const Glyph* findGlyph(char c) {
  for (const auto& g : kGlyphs) {
    if (g.ch == c) return &g;
  }
  return nullptr;
}

}  // namespace

const uint8_t* glyphRows(char c) {
  const Glyph* g = findGlyph(c);
  return g ? g->rows : kFallback;
}

bool glyphSupported(char c) { return findGlyph(c) != nullptr; }

int textWidth(const std::string& text) {
  if (text.empty()) return 0;
  return static_cast<int>(text.size()) * 4 - 1;  // 3 px glyph + 1 px spacing
}

}  // namespace sir
