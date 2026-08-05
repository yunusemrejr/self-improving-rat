#include "rendering/renderer.h"

#include "utility/bitmap_font.h"

#include <algorithm>
#include <cstdio>

namespace sir {

namespace {

// Monochrome palette (restrained, no bright colors).
constexpr SDL_Color kBg = {0x14, 0x14, 0x14, 0xFF};      // charcoal
constexpr SDL_Color kWall = {0x3a, 0x3a, 0x3a, 0xFF};    // dark gray
constexpr SDL_Color kFloor = {0x56, 0x56, 0x56, 0xFF};   // medium gray
constexpr SDL_Color kRat = {0xd8, 0xd8, 0xd8, 0xFF};     // light gray
constexpr SDL_Color kRatRest = {0x9a, 0x9a, 0x9a, 0xFF};
constexpr SDL_Color kNose = {0x14, 0x14, 0x14, 0xFF};
constexpr SDL_Color kCheese = {0xb0, 0xb0, 0xb0, 0xFF};  // muted off-white
constexpr SDL_Color kCheeseNotch = {0x8c, 0x8c, 0x8c, 0xFF};
constexpr SDL_Color kText = {0x99, 0x99, 0x99, 0xFF};
constexpr SDL_Color kTextHi = {0xcc, 0xcc, 0xcc, 0xFF};
constexpr SDL_Color kPanelBg = {0x1c, 0x1c, 0x1c, 0xFF};

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
  tile_ = cfg.tile_size;
  maze_px_w_ = cfg.maze_width * tile_;
  maze_px_h_ = cfg.maze_height * tile_;
  panel_x_ = maze_x_ + maze_px_w_ + 16;

  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    if (err) *err = std::string("SDL_Init failed: ") + SDL_GetError();
    return false;
  }
  win_ = SDL_CreateWindow("Self Improving Rat", SDL_WINDOWPOS_CENTERED,
                          SDL_WINDOWPOS_CENTERED, cfg.window_width,
                          cfg.window_height, SDL_WINDOW_SHOWN);
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
  for (int y = 0; y < maze.height(); ++y) {
    for (int x = 0; x < maze.width(); ++x) {
      const bool wall = maze.isWall(x, y);
      fillRect(maze_x_ + x * tile_, maze_y_ + y * tile_, tile_, tile_,
               wall ? kWall.r : kFloor.r, wall ? kWall.g : kFloor.g,
               wall ? kWall.b : kFloor.b);
    }
  }
}

void Renderer::drawCheese(const Simulation& sim) {
  for (const Position& c : sim.cheeses()) {
    const int px = maze_x_ + c.x * tile_;
    const int py = maze_y_ + c.y * tile_;
    const int body = std::max(4, tile_ - 5);
    // Cheese block with a bite (notch) in the top-right corner.
    fillRect(px + 2, py + 2, body, body, kCheese.r, kCheese.g, kCheese.b);
    const int notch = std::max(2, body / 3);
    fillRect(px + 2 + body - notch, py + 2, notch, notch, kCheeseNotch.r,
             kCheeseNotch.g, kCheeseNotch.b);
  }
}

void Renderer::drawRat(const Simulation& sim, bool resting) {
  const Rat& rat = sim.rat();
  const int px = maze_x_ + rat.position().x * tile_;
  const int py = maze_y_ + rat.position().y * tile_;

  const int body = std::max(4, tile_ - 7);
  const int bx = px + (tile_ - body) / 2;
  const int by = py + (tile_ - body) / 2;

  // Resting (consolidation): dimmed, curled body without nose/feet. The
  // visual reflects a real consolidation operation in progress.
  if (resting) {
    fillRect(bx + 1, by + 1, body - 2, body - 2, kRatRest.r, kRatRest.g,
             kRatRest.b);
    return;
  }

  // Body.
  fillRect(bx, by, body, body, kRat.r, kRat.g, kRat.b);

  // Nose in the facing direction + tiny walking feet (2-frame animation).
  const int off = rat.walkFrame() * 1;
  switch (rat.facing()) {
    case Action::Up:
      fillRect(bx + body / 2 - 1, by - 2, 2, 2, kNose.r, kNose.g, kNose.b);
      fillRect(bx + 1, by + body - 2 + off, 2, 2, kRat.r, kRat.g, kRat.b);
      fillRect(bx + body - 3, by + body - 2 - off, 2, 2, kRat.r, kRat.g, kRat.b);
      break;
    case Action::Down:
      fillRect(bx + body / 2 - 1, by + body, 2, 2, kNose.r, kNose.g, kNose.b);
      fillRect(bx + 1, by + off, 2, 2, kRat.r, kRat.g, kRat.b);
      fillRect(bx + body - 3, by - off, 2, 2, kRat.r, kRat.g, kRat.b);
      break;
    case Action::Left:
      fillRect(bx - 2, by + body / 2 - 1, 2, 2, kNose.r, kNose.g, kNose.b);
      fillRect(bx + 1 + off, by + 1, 2, 2, kRat.r, kRat.g, kRat.b);
      fillRect(bx + body - 3 - off, by + body - 3, 2, 2, kRat.r, kRat.g, kRat.b);
      break;
    case Action::Right:
      fillRect(bx + body, by + body / 2 - 1, 2, 2, kNose.r, kNose.g, kNose.b);
      fillRect(bx + 1 - off, by + 1, 2, 2, kRat.r, kRat.g, kRat.b);
      fillRect(bx + body - 3 + off, by + body - 3, 2, 2, kRat.r, kRat.g, kRat.b);
      break;
    default:
      break;
  }
}

void Renderer::drawStatusBar(const PanelData& p) {
  std::string status = p.paused ? "PAUSED" : (p.consolidating ? "REST" : "RUNNING");
  std::string line = "RAT  S " + formatInt(static_cast<int64_t>(p.lifetime_steps)) +
                     "  CH " + formatInt(static_cast<int64_t>(p.cheese_total)) +
                     "  E " + formatFloat(p.avg_energy, 2) +
                     "  H " + formatFloat(p.avg_hunger, 2) +
                     "  F " + formatFloat(p.avg_fatigue, 2) +
                     "  S " + formatFloat(p.avg_stress, 2) +
                     "  [ " + status + " ]";
  drawText(8, 8, line, 2, kTextHi.r, kTextHi.g, kTextHi.b);
}

void Renderer::drawPanel(const PanelData& p) {
  if (!p.debug) return;
  const int x = panel_x_;
  int y = 8;
  const int scale = 2;
  auto row = [&](const std::string& text, bool hi = false) {
    drawText(x, y, text, scale, hi ? kTextHi.r : kText.r,
             hi ? kTextHi.g : kText.g, hi ? kTextHi.b : kText.b);
    y += 14;
  };
  row("MAZE " + formatInt(p.maze_generations) + "  CHEESE " + formatInt(p.cheese_total), true);
  row("AGE " + formatDuration(p.runtime_s) + "  STEPS " + formatInt(p.lifetime_steps));
  row("EPISODES " + formatInt(static_cast<int64_t>(p.episode_count)) + "  STEPS/CH " +
          formatFloat(p.mean_steps, 0) + " med " + formatFloat(p.median_steps, 0));
  row("ADAPT " + formatFloat(p.adaptation_steps, 0) + " steps");
  row("WALL/1K " + formatFloat(p.wall_rate, 1) + "  OSC/1K " + formatFloat(p.revisit_rate, 1) +
      "  REW " + formatFloat(p.avg_reward, 3));
  row("DIV " + formatFloat(p.entropy, 2) + "  REP " + formatFloat(p.repeat_rate, 2) +
      "  EXP " + formatFloat(p.exploration_rate, 2) + "  EPS " + formatFloat(p.epsilon, 2));
  row("E " + formatFloat(p.avg_energy, 2) + "  H " + formatFloat(p.avg_hunger, 2) +
      "  F " + formatFloat(p.avg_fatigue, 2) + "  ST " + formatFloat(p.avg_stress, 2));
  row("MIN-E " + formatFloat(p.min_energy, 2) + "  H>0.8 " + formatFloat(p.extreme_hunger * 100, 0) + "%");
  row("NOV " + formatFloat(p.avg_novelty, 2) + "  CUR " + formatFloat(p.curiosity_rate, 3) +
      "  UNC " + formatFloat(p.uncertainty, 2) + "  PRED " + formatFloat(p.pred_loss, 3));
  row("STRESS-FREE " + formatInt(p.stress_free_steps) + " steps");
  row("TRAIN " + formatInt(static_cast<int64_t>(p.training_updates)) +
      "  INVALID " + formatInt(static_cast<int64_t>(p.invalid_updates)));
  row("CONN A " + formatInt(static_cast<int64_t>(p.active_conn)) + " D " +
      formatInt(static_cast<int64_t>(p.dormant_conn)) + "  P " +
      formatInt(static_cast<int64_t>(p.pruned)) + " R " +
      formatInt(static_cast<int64_t>(p.rewired)) + " OK " +
      formatInt(static_cast<int64_t>(p.struct_ok)) + " REJ " +
      formatInt(static_cast<int64_t>(p.struct_rejected)));
  row("EPISODIC " + formatInt(static_cast<int64_t>(p.episodic_used)) + "/" +
      formatInt(static_cast<int64_t>(p.episodic_cap)) + "  REPL " +
      formatInt(static_cast<int64_t>(p.episodic_replacements)));
  row("CONSOL " + formatInt(static_cast<int64_t>(p.consolidation_cycles)) +
      "  OPS " + formatInt(static_cast<int64_t>(p.consolidation_ops)));
  row("CKPT " + formatInt(static_cast<int64_t>(p.checkpoints_saved)) + "  " +
      formatFloat(p.fps, 0) + " fps  " + formatFloat(p.sim_steps_per_sec, 1) + " sps");
  row("SPACE PAUSE   R MAZE   S SAVE   D DEBUG   ESC QUIT");
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
