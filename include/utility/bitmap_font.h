#pragma once
// Tiny 3x5 bitmap font for the on-screen diagnostics panel. Monochrome,
// hand-encoded, no external font files. Rows are stored top to bottom, one
// byte per row, only the low 3 bits are used (bit 0 = left pixel).

#include <cstddef>
#include <cstdint>
#include <string>

namespace sir {

// Returns the 5 row bytes for a character (0 for unknown glyphs).
// Rows are indices into a table; see bitmap_font.cpp for the actual glyphs.
const uint8_t* glyphRows(char c);

// Width in pixels of a text at scale 1 (each glyph is 3 wide + 1 spacing).
int textWidth(const std::string& text);

// Characters covered by the font (used by tests).
bool glyphSupported(char c);

}  // namespace sir
