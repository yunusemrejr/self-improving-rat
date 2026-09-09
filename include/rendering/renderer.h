#pragma once
// Software-rendered habitat with grey rat and golden cheese illustrations.

#include "simulation/simulation.h"
#include "utility/config.h"
#include "rendering/learning_notes.h"
#include "rendering/note_typesetting.h"

#include <SDL.h>

#include <cstdint>
#include <string>

namespace sir {

// Snapshot of everything the panel shows (filled by the application from
// simulation, agent and metrics; all values are real measurements).
struct PanelData {
  bool paused = false;
  bool consolidating = false;
  bool debug = false;
  double fps = 0.0;
  double sim_steps_per_sec = 0.0;

  uint64_t lifetime_steps = 0;
  uint64_t cheese_total = 0;
  uint32_t maze_generations = 0;
  uint32_t checkpoints_saved = 0;
  uint64_t episode_count = 0;
  int steps_since_cheese = 0;

  double epsilon = 0.0;
  double avg_reward = 0.0;
  double wall_rate = 0.0;     // per 1000 steps
  double revisit_rate = 0.0;  // per 1000 steps
  double mean_steps = 0.0;
  double median_steps = 0.0;
  double adaptation_steps = 0.0;
  double entropy = 0.0;
  double repeat_rate = 0.0;
  double exploration_rate = 0.0;

  double pred_loss = 0.0;
  double uncertainty = 0.0;
  double avg_novelty = 0.0;
  double curiosity_rate = 0.0;  // avg curiosity reward per step

  double avg_energy = 0.0;
  double min_energy = 0.0;
  double avg_hunger = 0.0;
  double extreme_hunger = 0.0;  // fraction of window with hunger > 0.8
  double avg_fatigue = 0.0;
  double avg_stress = 0.0;
  int stress_free_steps = 0;

  uint64_t training_updates = 0;
  uint64_t invalid_updates = 0;
  uint64_t explored_total = 0;

  size_t replay_used = 0;
  size_t replay_cap = 0;
  size_t episodic_used = 0;
  size_t episodic_cap = 0;
  size_t episodic_replacements = 0;
  size_t active_conn = 0;
  size_t dormant_conn = 0;
  uint64_t pruned = 0;
  uint64_t rewired = 0;
  uint64_t struct_ok = 0;
  uint64_t struct_rejected = 0;
  uint64_t consolidation_cycles = 0;
  uint64_t consolidation_ops = 0;
  uint64_t runtime_s = 0;
};

class Renderer {
 public:
  bool init(const Config& cfg, std::string* err);
  void shutdown();
  void render(const Simulation& sim, const PanelData& panel);
  bool handleEvent(const SDL_Event& event);
  bool saveScreenshot(const std::string& path) const;

 private:
  void drawMaze(const Simulation& sim);
  void drawRat(const Simulation& sim, bool resting);
  void drawCheese(const Simulation& sim);
  void drawPanel(const PanelData& panel);
  void drawStatusBar(const PanelData& panel);
  void drawLearningNote(uint64_t now);
  void openLearningNote(uint64_t now);
  void drawNoteText(int x, int y, const std::string& text, SDL_Color color);
  void drawMath(int x, int y, const MathLayout& layout);
  void drawText(int x, int y, const std::string& text, int scale, uint8_t r,
                uint8_t g, uint8_t b);
  void fillRect(int x, int y, int w, int h, uint8_t r, uint8_t g, uint8_t b);

  SDL_Window* win_ = nullptr;
  SDL_Renderer* ren_ = nullptr;
  int width_ = 1060, height_ = 960;
  int maze_x_ = 28;
  int maze_y_ = 118;
  int panel_x_ = 0;
  int tile_ = 13;
  int maze_px_w_ = 0;
  int maze_px_h_ = 0;
  Config note_config_;
  LearningNoteDeck notes_{0};
  LearningNote note_;
  std::vector<std::string> note_title_, note_explanation_, note_notation_;
  MathLayout note_math_;
  int note_notation_x_ = 40;
  int notes_y_ = 0;
  double note_epsilon_ = 0;
  bool note_ready_ = false;
  SDL_Rect hold_button_{}, next_button_{};
};

}  // namespace sir
