#include "rendering/renderer.h"

#include "utility/bitmap_font.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <array>

namespace sir {

namespace {

// Warm habitat, neutral grey fur, and a single golden food accent.
constexpr SDL_Color kBg = {24, 31, 33, 255};
constexpr SDL_Color kWall = {42, 53, 55, 255};
constexpr SDL_Color kFloor = {219, 211, 190, 255};
constexpr SDL_Color kText = {164, 181, 180, 255};
constexpr SDL_Color kTextHi = {235, 237, 225, 255};
constexpr SDL_Color kGold = {244, 192, 74, 255};

void color(SDL_Renderer* r, SDL_Color c) {
  SDL_SetRenderDrawColor(r, c.r, c.g, c.b, c.a);
}
void ellipse(SDL_Renderer* r, int x, int y, int rx, int ry, SDL_Color c) {
  color(r, c);
  rx = std::max(1, rx); ry = std::max(1, ry);
  for (int dy = -ry; dy <= ry; ++dy) {
    const int dx = static_cast<int>(std::round(rx * std::sqrt(
        std::max(0.0, 1.0 - double(dy * dy) / double(ry * ry)))));
    SDL_RenderDrawLine(r, x - dx, y + dy, x + dx, y + dy);
  }
}
void rounded(SDL_Renderer* r, int x, int y, int w, int h, int radius, SDL_Color c) {
  radius = std::max(0, std::min({radius, w / 2, h / 2}));
  color(r, c);
  SDL_Rect core{x + radius, y, w - radius * 2, h};
  SDL_RenderFillRect(r, &core);
  core = {x, y + radius, w, h - radius * 2}; SDL_RenderFillRect(r, &core);
  for (int dx : {radius, w - radius - 1})
    for (int dy : {radius, h - radius - 1})
      ellipse(r, x + dx, y + dy, radius, radius, c);
}
void polygon(SDL_Renderer* r, const std::vector<SDL_Point>& points, SDL_Color c) {
  int top = points[0].y, bottom = top;
  for (auto p : points) { top = std::min(top, p.y); bottom = std::max(bottom, p.y); }
  color(r, c);
  for (int y = top; y <= bottom; ++y) {
    std::vector<int> cuts;
    for (size_t i = 0; i < points.size(); ++i) {
      auto a = points[i], b = points[(i + 1) % points.size()];
      if ((a.y <= y && b.y > y) || (b.y <= y && a.y > y))
        cuts.push_back(a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y));
    }
    std::sort(cuts.begin(), cuts.end());
    for (size_t i = 1; i < cuts.size(); i += 2)
      SDL_RenderDrawLine(r, cuts[i - 1], y, cuts[i], y);
  }
}

std::string formatInt(int64_t v) { return std::to_string(v); }

std::string formatFloat(double v, int decimals) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
  return buf;
}

std::string formatDuration(uint64_t seconds) {
  const uint64_t h = seconds / 3600, m = (seconds % 3600) / 60, s = seconds % 60;
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%llu:%02llu:%02llu",
                static_cast<unsigned long long>(h),
                static_cast<unsigned long long>(m),
                static_cast<unsigned long long>(s));
  return buf;
}

}  // namespace

bool Renderer::init(const Config& cfg, std::string* err) {
  width_ = std::max(cfg.window_width, cfg.maze_width + 56);
  height_ = std::max(cfg.window_height, cfg.maze_height + 216);
  // Fit all supported mazes into the actual window without clipping.
  maze_y_ = height_ < 400 ? 82 : 118;
  const int reserve = width_ >= 900 ? 350 : 0;
  tile_ = std::max(1, std::min({cfg.tile_size, (width_ - 56 - reserve) / cfg.maze_width,
                               (height_ - maze_y_ - 82) / cfg.maze_height}));
  maze_px_w_ = cfg.maze_width * tile_;
  maze_px_h_ = cfg.maze_height * tile_;
  panel_x_ = maze_x_ + maze_px_w_ + 30;

  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    if (err) *err = std::string("SDL_Init failed: ") + SDL_GetError();
    return false;
  }
  win_ = SDL_CreateWindow("Self Improving Rat", SDL_WINDOWPOS_CENTERED,
                          SDL_WINDOWPOS_CENTERED, width_,
                          height_, SDL_WINDOW_SHOWN);
  if (!win_) {
    if (err) *err = std::string("SDL_CreateWindow failed: ") + SDL_GetError();
    SDL_Quit();
    return false;
  }
  // Software renderer: no GPU required, crisp pixels, deterministic output.
  ren_ = SDL_CreateRenderer(win_, -1, SDL_RENDERER_SOFTWARE);
  if (!ren_) {
    if (err) *err = std::string("SDL_CreateRenderer failed: ") + SDL_GetError();
    SDL_DestroyWindow(win_);
    win_ = nullptr;
    SDL_Quit();
    return false;
  }
  SDL_SetRenderDrawBlendMode(ren_, SDL_BLENDMODE_NONE);
  return true;
}

void Renderer::shutdown() {
  if (ren_) SDL_DestroyRenderer(ren_);
  if (win_) SDL_DestroyWindow(win_);
  ren_ = nullptr;
  win_ = nullptr;
  SDL_Quit();
}

void Renderer::fillRect(int x, int y, int w, int h, uint8_t r, uint8_t g,
                        uint8_t b) {
  SDL_Rect rc{x, y, w, h};
  SDL_SetRenderDrawColor(ren_, r, g, b, 255);
  SDL_RenderFillRect(ren_, &rc);
}

void Renderer::drawText(int x, int y, const std::string& text, int scale,
                        uint8_t r, uint8_t g, uint8_t b) {
  SDL_SetRenderDrawColor(ren_, r, g, b, 255);
  int cx = x;
  for (char c : text) {
    const uint8_t* rows = glyphRows(c);
    for (int row = 0; row < 5; ++row) {
      uint8_t bits = rows[row];
      for (int col = 0; col < 3; ++col) {
        if (bits & (1u << (2 - col))) {
          SDL_Rect rc{cx + col * scale, y + row * scale, scale, scale};
          SDL_RenderFillRect(ren_, &rc);
        }
      }
    }
    cx += 4 * scale;
  }
}

void Renderer::drawMaze(const Simulation& sim) {
  const Maze& maze = sim.maze();
  rounded(ren_, maze_x_ - 10, maze_y_ - 10, maze_px_w_ + 20, maze_px_h_ + 20,
          16, kWall);
  const int inset = std::max(0, tile_ / 10);
  for (int y = 0; y < maze.height(); ++y) {
    for (int x = 0; x < maze.width(); ++x) {
      if (maze.isWall(x, y)) continue;
      const int px = maze_x_ + x * tile_, py = maze_y_ + y * tile_;
      rounded(ren_, px + inset, py + inset, tile_ - 2 * inset, tile_ - 2 * inset,
              tile_ / 5, kFloor);
      if (!maze.isWall(x + 1, y))
        fillRect(px + tile_ / 2, py + inset, tile_, tile_ - 2 * inset,
                 kFloor.r, kFloor.g, kFloor.b);
      if (!maze.isWall(x, y + 1))
        fillRect(px + inset, py + tile_ / 2, tile_ - 2 * inset, tile_,
                 kFloor.r, kFloor.g, kFloor.b);
    }
  }
}

void Renderer::drawCheese(const Simulation& sim) {
  for (const Position& c : sim.cheeses()) {
    const float unit = tile_ / 40.0f;
    const int cx = maze_x_ + c.x * tile_ + tile_ / 2;
    const int cy = maze_y_ + c.y * tile_ + tile_ / 2;
    auto p = [&](float x, float y) { return SDL_Point{cx + int(x * unit), cy + int(y * unit)}; };
    auto hole = [&](float x, float y, float rx, float ry) {
      auto q = p(x, y);
      ellipse(ren_, q.x, q.y, int(rx * unit), int(ry * unit), {190, 119, 24, 255});
      ellipse(ren_, q.x + std::max(1, int(unit)), q.y + std::max(1, int(unit)),
              int(rx * unit * .65f), int(ry * unit * .55f), {238, 165, 43, 255});
    };
    ellipse(ren_, cx, cy + int(12 * unit), int(15 * unit), int(4 * unit), {167, 157, 136, 255});
    // Three faces of a Swiss-cheese wedge, including its amber rind.
    polygon(ren_, {p(-15,-4), p(5,-14), p(16,0), p(16,10), p(-15,11)}, {147, 96, 26, 255});
    polygon(ren_, {p(-14,-3), p(15,0), p(15,9), p(-14,10)}, {246, 181, 47, 255});
    polygon(ren_, {p(-14,-4), p(5,-13), p(15,-1)}, {255, 222, 106, 255});
    polygon(ren_, {p(5,-13), p(16,0), p(16,9), p(13,7), p(13,-1)}, {229, 153, 32, 255});
    hole(-7,3,2.7f,2.5f); hole(6,4,3,3); hole(0,-7,2.5f,1.5f);
  }
}

void Renderer::drawRat(const Simulation& sim, bool resting) {
  const Rat& rat = sim.rat();
  const int cx = maze_x_ + rat.position().x * tile_ + tile_ / 2;
  const int cy = maze_y_ + rat.position().y * tile_ + tile_ / 2;
  const float u = tile_ / 40.0f;
  const int dir = resting ? 3 : static_cast<int>(rat.facing());
  auto point = [&](float x, float y) {
    if (dir == 0) { const float t=x; x=y; y=-t; }
    else if (dir == 1) { const float t=x; x=-y; y=t; }
    else if (dir == 2) { x=-x; y=-y; }
    return SDL_Point{cx + int(std::round(x*u)), cy + int(std::round(y*u))};
  };
  auto oval = [&](float x, float y, float rx, float ry, SDL_Color c) {
    auto q=point(x,y);
    if (dir < 2) std::swap(rx, ry);
    ellipse(ren_, q.x, q.y, int(rx*u), int(ry*u), c);
  };
  auto line = [&](float x, float y, float xx, float yy, SDL_Color c) {
    color(ren_,c); auto a=point(x,y), b=point(xx,yy);
    SDL_RenderDrawLine(ren_,a.x,a.y,b.x,b.y);
  };
  const SDL_Color outline{67,71,73,255}, fur{139,144,147,255}, light{175,180,182,255};
  const SDL_Color pink{191,140,136,255};
  oval(-1,3,14,10,{170,161,143,255});
  // Long, tapering pink tail. Distinct from the pointed muzzle.
  for (int t=0;t<16;++t) {
    const float x=-10-t*.48f, y=3+std::sin(t*.23f)*6;
    oval(x,y,t<7?1.6f:1.0f,t<7?1.6f:1.0f,pink);
  }
  const float walk = resting ? 0 : (rat.walkFrame() ? 1.2f : -1.2f);
  oval(-6+walk,-8,3,2,pink); oval(-6-walk,8,3,2,pink);
  oval(6-walk,-5,2,2,pink); oval(6+walk,5,2,2,pink);
  oval(-3,0,12,9,outline); oval(-3,-.5f,11,8,fur); oval(-5,-2,7,5,light);
  polygon(ren_,{point(3,-7),point(15,-2),point(17,0),point(15,3),point(3,7)},outline);
  polygon(ren_,{point(3,-6),point(15,-1),point(16,0),point(14,2),point(3,6)},fur);
  oval(4,-6,4.5f,4.5f,outline); oval(4,-6,3.5f,3.5f,light); oval(4,-6,2.2f,2.2f,pink);
  oval(4,6,4,4,outline); oval(4,6,3,3,fur); oval(4,6,1.9f,1.9f,pink);
  if (resting) line(10,-2,13,-2,outline);
  else { oval(11,-2.5f,1.7f,1.7f,{24,29,31,255}); oval(11,-3,0.6f,0.6f,{245,246,238,255}); }
  oval(16,0,1.6f,1.8f,pink);
  line(13,-1,18,-6,{97,97,91,255}); line(13,1,19,5,{97,97,91,255});
  line(13,-1,19,-3,{97,97,91,255}); line(13,1,18,7,{97,97,91,255});
}

void Renderer::drawStatusBar(const PanelData& p) {
  const int scale = width_ >= 600 ? 3 : 2;
  drawText(28, height_ < 400 ? 16 : 26, "SELF IMPROVING RAT", scale, kTextHi.r, kTextHi.g, kTextHi.b);
  drawText(28, height_ < 400 ? 38 : 54, "A LITTLE LIFE. A LIFETIME OF LEARNING.", width_ >= 600 ? 2 : 1,
           kText.r, kText.g, kText.b);
  const std::string status = p.paused ? "PAUSED" : p.consolidating ? "RESTING + LEARNING" : "EXPLORING";
  drawText(28, height_ < 400 ? 58 : 85, status, 2, kGold.r, kGold.g, kGold.b);
  drawText(28, height_ - 48, "SPACE PAUSE   R NEW MAZE   S SAVE", width_ >= 600 ? 2 : 1,
           kText.r, kText.g, kText.b);
  drawText(28, height_ - 28, "D DIAGNOSTICS   ESC SAVE + EXIT", width_ >= 600 ? 2 : 1,
           kText.r, kText.g, kText.b);
}

void Renderer::drawPanel(const PanelData& p) {
  if (panel_x_ + 286 > width_) {
    drawText(28, height_ - 72, "CHEESE " + formatInt(p.cheese_total) +
             "  STEPS " + formatInt(p.lifetime_steps), width_ >= 600 ? 2 : 1,
             kTextHi.r, kTextHi.g, kTextHi.b);
    return;
  }
  const int x = panel_x_;
  int y = 120;
  auto row = [&](const std::string& text, bool hi = false) {
    if (y > height_ - 75) return;
    drawText(x, y, text, 2, hi ? kTextHi.r : kText.r,
             hi ? kTextHi.g : kText.g, hi ? kTextHi.b : kText.b);
    y += 23;
  };
  if (p.debug) {
    row("TRAINING DIAGNOSTICS", true); y += 6;
    row("CHEESE " + formatInt(p.cheese_total) + "  MAZE " + formatInt(p.maze_generations));
    row("STEPS " + formatInt(p.lifetime_steps));
    row("UPDATES " + formatInt(p.training_updates));
    row("INVALID UPDATES " + formatInt(p.invalid_updates));
    row("EPSILON " + formatFloat(p.epsilon, 3));
    row("PRED LOSS " + formatFloat(p.pred_loss, 3));
    row("NOVELTY " + formatFloat(p.avg_novelty, 3));
    row("REWARD " + formatFloat(p.avg_reward, 3));
    row("WALL / 1K " + formatFloat(p.wall_rate, 1));
    row("REVISIT / 1K " + formatFloat(p.revisit_rate, 1));
    row("STEPS / CHEESE " + formatFloat(p.mean_steps, 0));
    row("MEMORIES " + formatInt(p.episodic_used) + "  REPLAY " + formatInt(p.replay_used));
    row("CONNECTIONS " + formatInt(p.active_conn));
    row("PRUNED " + formatInt(p.pruned) + "  REWIRED " + formatInt(p.rewired));
    row("RESTS " + formatInt(p.consolidation_cycles) + "  OPS " + formatInt(p.consolidation_ops));
    row("SAVED " + formatInt(p.checkpoints_saved) + "  " + formatFloat(p.fps, 0) + " FPS");
    row("D TO RETURN TO ORGANISM");
    return;
  }
  auto meter = [&](const std::string& label, double value, SDL_Color c) {
    row(label + "  " + formatFloat(value * 100, 0) + "%");
    rounded(ren_, x, y-5, 260, 5, 2, kWall);
    const int w = int(260 * std::clamp(value, 0.0, 1.0));
    if (w > 0) rounded(ren_, x, y-5, w, 5, 2, c);
    y += 16;
  };
  row("LIFETIME", true); y += 6;
  row(formatInt(p.cheese_total) + " CHEESE FOUND", true);
  row(formatInt(p.lifetime_steps) + " STEPS  /  MAZE " + formatInt(p.maze_generations));
  row("AWAKE " + formatDuration(p.runtime_s)); y += 15;
  meter("ENERGY", p.avg_energy, {153,187,160,255});
  meter("HUNGER", p.avg_hunger, kGold);
  meter("FATIGUE", p.avg_fatigue, {152,169,187,255});
  y += 12; row("LEARNING", true);
  row(formatInt(p.training_updates) + " NEURAL UPDATES");
  row("MEMORIES " + formatInt(p.episodic_used) + "/" + formatInt(p.episodic_cap));
  row("REPLAY " + formatInt(p.replay_used) + "/" + formatInt(p.replay_cap));
  row("SAVED " + formatInt(p.checkpoints_saved) + "  RESTS " + formatInt(p.consolidation_cycles));
  row("D FOR TRAINING DIAGNOSTICS");
}

void Renderer::render(const Simulation& sim, const PanelData& panel) {
  SDL_SetRenderDrawColor(ren_, kBg.r, kBg.g, kBg.b, 255);
  SDL_RenderClear(ren_);

  drawMaze(sim);
  drawCheese(sim);
  drawRat(sim, panel.consolidating);
  drawStatusBar(panel);
  drawPanel(panel);

  SDL_RenderPresent(ren_);
}

bool Renderer::saveScreenshot(const std::string& path) const {
  if (!ren_) return false;
  int w, h;
  SDL_GetRendererOutputSize(ren_, &w, &h);
  SDL_Surface* surf =
      SDL_CreateRGBSurfaceWithFormat(0, w, h, 32, SDL_PIXELFORMAT_ARGB8888);
  if (!surf) return false;
  if (SDL_RenderReadPixels(ren_, nullptr, SDL_PIXELFORMAT_ARGB8888, surf->pixels,
                           surf->pitch) != 0) {
    SDL_FreeSurface(surf);
    return false;
  }
  const int rc = SDL_SaveBMP(surf, path.c_str());
  SDL_FreeSurface(surf);
  return rc == 0;
}

}  // namespace sir
