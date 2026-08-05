#include "utility/rolling_stats.h"

#include <algorithm>
#include <cmath>

namespace sir {

Metrics::Metrics(size_t step_window, size_t episode_window, size_t adapt_window)
    : reward_w_(step_window),
      wall_w_(step_window),
      revisit_w_(step_window),
      energy_w_(step_window),
      hunger_w_(step_window),
      fatigue_w_(step_window),
      stress_w_(step_window),
      novelty_w_(step_window),
      curiosity_w_(step_window),
      predloss_w_(step_window),
      actions_(step_window),
      explore_w_(step_window),
      repeat_w_(step_window),
      episode_steps_(episode_window),
      adaptation_w_(adapt_window) {}

void Metrics::recordStep(float reward, bool wall_hit, bool revisit, int action,
                         int prev_action, bool explored,
                         const Homeostasis::Snapshot& h, float novelty,
                         float curiosity_reward, float pred_loss) {
  reward_w_.push(reward);
  wall_w_.push(wall_hit ? 1.0 : 0.0);
  revisit_w_.push(revisit ? 1.0 : 0.0);
  energy_w_.push(h.energy);
  hunger_w_.push(h.hunger);
  fatigue_w_.push(h.fatigue);
  stress_w_.push(h.stress);
  novelty_w_.push(novelty);
  curiosity_w_.push(curiosity_reward);
  predloss_w_.push(pred_loss);
  actions_.push(action);
  explore_w_.push(explored ? 1.0 : 0.0);
  repeat_w_.push(action == prev_action ? 1.0 : 0.0);
  if (wall_hit) ++wall_hits_total_;
}

void Metrics::recordCheese(int steps_this_episode) {
  episode_steps_.push(static_cast<double>(steps_this_episode));
}

void Metrics::recordAdaptation(int steps) {
  adaptation_w_.push(static_cast<double>(steps));
}

double Metrics::recentAvgReward() const {
  return reward_w_.empty() ? 0.0
                           : reward_w_.sum() / static_cast<double>(reward_w_.size());
}

double Metrics::wallRatePer1000Steps() const {
  return wall_w_.empty()
             ? 0.0
             : wall_w_.sum() * 1000.0 / static_cast<double>(wall_w_.size());
}

double Metrics::revisitRatePer1000Steps() const {
  return revisit_w_.empty()
             ? 0.0
             : revisit_w_.sum() * 1000.0 / static_cast<double>(revisit_w_.size());
}

double Metrics::meanStepsPerCheese() const {
  if (episode_steps_.empty()) return 0.0;
  return episode_steps_.sum() / static_cast<double>(episode_steps_.size());
}

double Metrics::medianStepsPerCheese() const {
  if (episode_steps_.empty()) return 0.0;
  std::vector<double> v;
  v.reserve(episode_steps_.size());
  for (size_t i = 0; i < episode_steps_.size(); ++i)
    v.push_back(episode_steps_[i]);
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

double Metrics::meanAdaptationSteps() const {
  if (adaptation_w_.empty()) return 0.0;
  return adaptation_w_.sum() / static_cast<double>(adaptation_w_.size());
}

double Metrics::avgEnergy() const {
  return energy_w_.empty() ? 0.0
                           : energy_w_.sum() / static_cast<double>(energy_w_.size());
}

double Metrics::minEnergy() const {
  if (energy_w_.empty()) return 0.0;
  double m = energy_w_[0];
  for (size_t i = 1; i < energy_w_.size(); ++i) m = std::min(m, energy_w_[i]);
  return m;
}

double Metrics::avgHunger() const {
  return hunger_w_.empty() ? 0.0
                           : hunger_w_.sum() / static_cast<double>(hunger_w_.size());
}

double Metrics::extremeHungerFraction() const {
  if (hunger_w_.empty()) return 0.0;
  size_t n = 0;
  for (size_t i = 0; i < hunger_w_.size(); ++i)
    if (hunger_w_[i] > 0.8) ++n;
  return static_cast<double>(n) / static_cast<double>(hunger_w_.size());
}

double Metrics::avgFatigue() const {
  return fatigue_w_.empty()
             ? 0.0
             : fatigue_w_.sum() / static_cast<double>(fatigue_w_.size());
}

double Metrics::avgStress() const {
  return stress_w_.empty()
             ? 0.0
             : stress_w_.sum() / static_cast<double>(stress_w_.size());
}

double Metrics::avgNovelty() const {
  return novelty_w_.empty()
             ? 0.0
             : novelty_w_.sum() / static_cast<double>(novelty_w_.size());
}

double Metrics::avgCuriosityReward() const {
  return curiosity_w_.empty()
             ? 0.0
             : curiosity_w_.sum() / static_cast<double>(curiosity_w_.size());
}

double Metrics::avgPredLoss() const {
  return predloss_w_.empty()
             ? 0.0
             : predloss_w_.sum() / static_cast<double>(predloss_w_.size());
}

double Metrics::actionEntropy() const {
  const size_t n = actions_.size();
  if (n == 0) return 0.0;
  double hist[4] = {0, 0, 0, 0};
  for (size_t i = 0; i < n; ++i) hist[actions_[i]] += 1.0;
  double h = 0.0;
  for (int k = 0; k < 4; ++k) {
    const double p = hist[k] / static_cast<double>(n);
    if (p > 0.0) h -= p * std::log(p);
  }
  return h / std::log(4.0);  // normalized to [0,1]
}

double Metrics::repeatedActionRate() const {
  return repeat_w_.empty()
             ? 0.0
             : repeat_w_.sum() / static_cast<double>(repeat_w_.size());
}

double Metrics::explorationRate() const {
  return explore_w_.empty()
             ? 0.0
             : explore_w_.sum() / static_cast<double>(explore_w_.size());
}

}  // namespace sir
