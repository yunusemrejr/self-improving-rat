#pragma once

#include <array>
#include <string>
#include <vector>

namespace sir {

// A small, original bitmap alphabet for readable sentence case and mathematics.
// Nine rows include room for lowercase descenders. No font download is needed.
using NoteGlyph = std::array<unsigned char, 9>;
const NoteGlyph* noteGlyph(char32_t character);

struct MathMark {
  char32_t glyph = 0; // zero means a line from (x,y) to (x2,y2)
  int x = 0, y = 0, x2 = 0, y2 = 0, scale = 1;
};
struct MathLayout {
  int width = 0, height = 0, baseline = 0;
  std::vector<MathMark> marks;
};

// Typesets the curated notes' LaTeX subset, not arbitrary TeX. Unsupported or
// malformed input throws rather than silently displaying a misleading formula.
MathLayout typesetNoteMath(const std::string& latex, int scale = 3);

} // namespace sir
