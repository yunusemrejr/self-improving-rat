#pragma once
// Bounded rolling-window statistics. All values derive from real simulation
// events and neural activity; nothing is fabricated or manually improved.

#include "organism/homeostasis.h"
#include "utility/ring_buffer.h"

#include <cstdint>
#include <vector>

namespace sir {

class Metrics {
 public:
  Metrics(size_t step_window = 2000, size_t episode_window = 200,
          size_t adapt_window = 20);

  // One simulation step: reward, events, action (for diversity), exploration
  // flag, homeostatic snapshot, novelty, curiosity reward, prediction loss.
  void recordStep(float reward, bool wall_hit, bool revisit, int action,
                  int prev_action, bool explored,
                  const Homeostasis::Snapshot& h, float novelty,
                  float curiosity_reward, float pred_loss);

  void recordCheese(int steps_this_episode);
  // Steps from a maze regeneration to the next cheese (adaptation speed).
  void recordAdaptation(int steps);

  uint64_t wallHitsTotal() const { return wall_hits_total_; }
  void setWallHitsTotal(uint64_t v) { wall_hits_total_ = v; }

  // --- navigation windows ---
  double recentAvgReward() const;
  double wallRatePer1000Steps() const;
  double revisitRatePer1000Steps() const;
  double meanStepsPerCheese() const;
  double medianStepsPerCheese() const;
  size_t episodeCount() const { return episode_steps_.size(); }
  double meanAdaptationSteps() const;

  // --- organism windows ---
  double avgEnergy() const;
  double minEnergy() const;
  double avgHunger() const;
  double extremeHungerFraction() const;  // fraction of window with hunger > 0.8
  double avgFatigue() const;
  double avgStress() const;
  double avgNovelty() const;
  double avgCuriosityReward() const;
  double avgPredLoss() const;

  // --- behavioral diversity ---
  double actionEntropy() const;        // normalized [0,1] over the window
  double repeatedActionRate() const;   // action == previous action
  double explorationRate() const;      // fraction of exploratory draws

 private:
  uint64_t wall_hits_total_ = 0;

  RingBuffer<double> reward_w_;
  RingBuffer<double> wall_w_;
  RingBuffer<double> revisit_w_;
  RingBuffer<double> energy_w_;
  RingBuffer<double> hunger_w_;
  RingBuffer<double> fatigue_w_;
  RingBuffer<double> stress_w_;
  RingBuffer<double> novelty_w_;
  RingBuffer<double> curiosity_w_;
  RingBuffer<double> predloss_w_;
  RingBuffer<int> actions_;        // action ids (entropy/repetition)
  RingBuffer<double> explore_w_;   // 1 = exploratory draw
  RingBuffer<double> repeat_w_;    // 1 = action repeated previous action
  RingBuffer<double> episode_steps_;
  RingBuffer<double> adaptation_w_;
};

}  // namespace sir
