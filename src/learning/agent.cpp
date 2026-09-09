#include "learning/agent.h"

#include <algorithm>
#include <cmath>

namespace sir {

namespace {

inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace

Agent::Agent(const Config& cfg, Rng& rng)
    : cfg_(cfg),
      rng_(rng),
      development_(DevParams{cfg.maturity_steps, cfg.lr_age_scale,
                             cfg.epsilon_age_scale, cfg.plasticity_age_scale}),
      online_(kObservationBase * cfg.observation_frames, cfg.rnn_hidden, 4,
              predOutStatic(), rng),
      target_(kObservationBase * cfg.observation_frames, cfg.rnn_hidden, 4,
              predOutStatic(), rng),
      replay_(kObservationBase * cfg.observation_frames, cfg.rnn_hidden,
              static_cast<size_t>(cfg.replay_capacity), static_cast<float>(cfg.per_priority_alpha)),
      episodic_(kObservationBase * cfg.observation_frames, cfg.rnn_hidden,
                static_cast<size_t>(cfg.episodic_memory_capacity)),
      novelty_(static_cast<size_t>(cfg.novelty_table_capacity)),
      h_(cfg.rnn_hidden, 0.0f),
      last_pred_(predOutStatic(), 0.0f),
      input_size_(kObservationBase * cfg.observation_frames) {
  const size_t n = online_.paramCount();
  adam_m_.assign(n, 0.0f);
  adam_v_.assign(n, 0.0f);
  // Scratch sizing (fixed for the agent's lifetime; see agent.h).
  action_visits_.assign(static_cast<size_t>(cfg.maze_width) * cfg.maze_height * 4, 0);
  sample_indices_.reserve(cfg.batch_size);
  burn_h_.resize(cfg.rnn_hidden); burn_ht_.resize(cfg.rnn_hidden);
  burn_next_.resize(cfg.rnn_hidden); burn_next_t_.resize(cfg.rnn_hidden);
  seq_target_chain_.resize((kSeqChunkLen + 1) * cfg.rnn_hidden);
  scratch_h2_.resize(cfg.rnn_hidden);
  scratch_q_.resize(policyOut());
  tb_h_.resize(cfg.rnn_hidden);
  tb_q_.resize(policyOut());
  tb_pred_.resize(predOut());
  tb_h2_.resize(cfg.rnn_hidden);
  tb_q2_.resize(policyOut());
  tb_h2t_.resize(cfg.rnn_hidden);
  tb_q2t_.resize(policyOut());
  tb_grad_q_.resize(policyOut());
  tb_grad_p_.resize(predOut());
  tb_grad_.assign(n, 0.0f);
  tb_gb_.resize(n);
  tb_snapshot_.resize(n);
  adam_params_.resize(n);
  soft_po_.resize(n);
  soft_pt_.resize(n);
  seq_hchain_.resize((kSeqChunkLen + 1) * cfg.rnn_hidden);
  seq_q_.resize(kSeqChunkLen * policyOut());
  seq_pred_.resize(kSeqChunkLen * predOut());
  seq_grad_q_.resize(kSeqChunkLen * policyOut());
  seq_grad_pred_.resize(kSeqChunkLen * predOut());
  seq_hp_a_.resize(cfg.rnn_hidden);
  seq_hp_b_.resize(cfg.rnn_hidden);
  seq_gb_a_.resize(n);
  seq_grad_total_.resize(n);
  seq_snapshot_.resize(n);
  seq_h2_.resize(cfg.rnn_hidden);
  seq_q2_.resize(policyOut());
  seq_h2t_.resize(cfg.rnn_hidden);
  seq_q2t_.resize(policyOut());
  seq_t_.resize(predOut());
  epsilon_ = cfg.exploration_start;
  // The target network starts as a copy of the online network.
  std::vector<float> p(n);
  online_.getParams(p.data());
  target_.setParams(p.data());
}

float Agent::computeEpsilon(uint64_t steps) const {
  double e = cfg_.exploration_end +
             (cfg_.exploration_start - cfg_.exploration_end) *
                 std::max(0.0, 1.0 - static_cast<double>(steps) /
                                          std::max(1, cfg_.exploration_decay_steps));
  e *= development_.epsilonScale(steps);
  e = std::max(e, cfg_.exploration_end);
  return static_cast<float>(clampf(e, 0.0, 1.0));
}

void Agent::qValuesFor(const float* obs, float* q_out) const {
  std::vector<float> h2(rnnSize());
  // Do not mutate the persistent recurrent state; a local copy is enough.
  online_.forward(obs, h_.data(), h2.data(), q_out, nullptr);
}

void Agent::resetRecurrent() {
  std::fill(h_.begin(), h_.end(), 0.0f);
  replay_.markBoundary();  // truncate sequence flow, but retain value bootstrap
  std::fill(action_visits_.begin(), action_visits_.end(), 0);
}

int Agent::greedyAction(const float* obs, const float* q) const {
  constexpr int wall[4] = {0, 4, 6, 2}; // Up, Down, Left, Right
  const float* frame = obs + input_size_ - kObservationBase;
  int best = -1;
  for (int k = 0; k < 4; ++k) {
    if (cfg_.mask_wall_actions && frame[wall[k]] >= 0.5f) continue;
    if (best < 0 || q[k] > q[best]) best = k;
  }
  // Defensive fallback for impossible/synthetic all-blocked observations.
  if (best < 0) best = static_cast<int>(std::max_element(q, q + 4) - q);
  return best;
}

void Agent::burnIn(size_t start, float* online_h, float* target_h) {
  size_t first = start;
  for (int i = 0; i < cfg_.replay_burn_in && first > 0 && !replay_.boundary(first - 1); ++i)
    --first;
  std::copy(replay_.h(first), replay_.h(first) + rnnSize(), online_h);
  std::copy(replay_.h(first), replay_.h(first) + rnnSize(), target_h);
  for (size_t i = first; i < start; ++i) {
    online_.forward(replay_.obs(i), online_h, burn_next_.data(), nullptr, nullptr);
    target_.forward(replay_.obs(i), target_h, burn_next_t_.data(), nullptr, nullptr);
    std::copy(burn_next_.begin(), burn_next_.end(), online_h);
    std::copy(burn_next_t_.begin(), burn_next_t_.end(), target_h);
  }
}

Agent::Decision Agent::selectAction(const float* obs, bool explore) {
  epsilon_ = computeEpsilon(lifetime_steps_);
  // Reused scratch (see agent.h); sizes are fixed at construction.
  std::vector<float>& h2 = scratch_h2_;
  std::vector<float>& q = scratch_q_;
  // h_out must not alias h_prev (see GruLayer).
  online_.forward(obs, h_.data(), h2.data(), q.data(), last_pred_.data());
  h_ = h2;

  if (!explore) return {static_cast<Action>(greedyAction(obs, q.data())), false};
  // Episodic count bonuses use only the rat's observed body position and
  // attempted actions. No map, cheese coordinates, or search algorithm.
  const float* frame = obs + input_size_ - kObservationBase;
  const int px = static_cast<int>(std::round(std::clamp(frame[28], 0.0f, 1.0f) * (cfg_.maze_width-1)));
  const int py = static_cast<int>(std::round(std::clamp(frame[29], 0.0f, 1.0f) * (cfg_.maze_height-1)));
  const size_t base = (static_cast<size_t>(py) * cfg_.maze_width + px) * 4;
  constexpr int wall_bits[4] = {0, 4, 6, 2};
  float minimum = 0, maximum = 0;
  uint64_t total_visits = 0;
  bool first = true;
  for (int k = 0; k < 4; ++k) {
    if (cfg_.mask_wall_actions && frame[wall_bits[k]] >= .5f) continue;
    if (first) { minimum = maximum = q[k]; first = false; }
    minimum = std::min(minimum, q[k]); maximum = std::max(maximum, q[k]);
    total_visits += action_visits_[base + k];
  }
  // UCB-style uncertainty grows for neglected actions as a state is revisited.
  // Scale legal Q differences to <=1 so a stale, overconfident prediction
  // cannot permanently overwhelm the exploration bonus in a loop.
  const float range = std::max(1.0f, maximum - minimum);
  const float log_visits = std::log(2.0f + static_cast<float>(total_visits));
  float scores[4];
  for (int k = 0; k < 4; ++k) {
    scores[k] = cfg_.episodic_action_bonus > 0 ?
        (q[k] - minimum) / range + static_cast<float>(cfg_.episodic_action_bonus) *
        std::sqrt(2.0f * log_visits / (1.0f + static_cast<float>(action_visits_[base + k]))) : q[k];
  }
  int chosen = greedyAction(obs, scores);
  const bool random = rng_.uniform01() < epsilon_;
  if (random) {
    constexpr int wall[4] = {0, 4, 6, 2};
    float weights[4]{}, total=0;
    for (int k=0;k<4;++k) {
      if (!cfg_.mask_wall_actions || frame[wall[k]] < .5f) {
        weights[k] = cfg_.episodic_action_bonus > 0 ?
            1.0f / std::sqrt(1.0f + static_cast<float>(action_visits_[base+k])) : 1.0f;
        total += weights[k];
      }
    }
    float draw = static_cast<float>(rng_.uniform01()) * total;
    for(int k=0;k<4;++k) { draw -= weights[k]; if(weights[k]>0 && draw<=0) { chosen=k; break; } }
    ++explored_count_;
  }
  auto& visits=action_visits_[base+chosen];
  if (visits < 1000000) ++visits;
  return {static_cast<Action>(chosen), random};
}

void Agent::updatePredictionTargets(const float* s, const float* s2,
                                    const float* homeo_targets, float reward_ext,
                                    float* out) const {
  s += input_size_ - kObservationBase;
  s2 += input_size_ - kObservationBase;
  // [0..11] next wall bits + next scent channels (directly observed).
  for (int k = 0; k < 12; ++k) out[k] = clampf(s2[k], 0.0f, 1.0f);
  // [12..14] homeostasis deltas mapped from [-1,1] to [0,1]. When the raw
  // deltas are unavailable (episodic replay), they are reconstructed from the
  // absolute homeostatic channels present in the observations.
  for (int k = 0; k < 3; ++k) {
    float d;
    if (homeo_targets) {
      d = homeo_targets[k];
    } else {
      d = s2[19 + k] - s[19 + k];
    }
    out[12 + k] = clampf((d + 1.0f) * 0.5f, 0.0f, 1.0f);
  }
  // [15] external reward mapped from [-12,12] to [0,1].
  out[15] = clampf((reward_ext + 12.0f) / 24.0f, 0.0f, 1.0f);
}

void Agent::computeIntrinsics(const float* s, const float* s2,
                              const float homeo_targets[3], float reward_ext) {
  // State novelty from the quantized observation code (decays on repeat
  // exposure by construction: novelty = 1/sqrt(count)).
  const float state_novelty = novelty_.observe(s, input_size_);

  // Prediction error against the observed next state (bounded, normalized).
  float t[16];
  updatePredictionTargets(s, s2, homeo_targets, reward_ext, t);
  float err = 0.0f;
  for (int k = 0; k < predOut(); ++k) {
    const float d = last_pred_[k] - t[k];
    err += d * d;
  }
  err /= static_cast<float>(predOut());
  last_pred_error_ = std::min(1.0f, err / 0.25f);
  const float pred_novelty = last_pred_error_;

  // Blended novelty, bounded to [0,1].
  last_novelty_ = clampf(
      static_cast<float>(cfg_.novelty_state_weight) * state_novelty +
          static_cast<float>(cfg_.novelty_pred_weight) * pred_novelty,
      0.0f, 1.0f);

  // Curiosity reward: small and capped at curiosity_reward_gain (much smaller
  // than the cheese reward). The application modulates it by the curiosity
  // need (homeostatic drive).
  last_curiosity_reward_ =
      static_cast<float>(cfg_.curiosity_reward_gain) * last_novelty_;

  // Uncertainty and prediction-loss EMAs.
  const float dec = static_cast<float>(cfg_.prediction_ema_decay);
  uncertainty_ = dec * uncertainty_ + (1.0f - dec) * last_pred_error_;
  pred_loss_ema_ = dec * pred_loss_ema_ + (1.0f - dec) * err;
  novelty_ema_ = dec * novelty_ema_ + (1.0f - dec) * last_novelty_;
}

void Agent::observeAndTrain(const float* s, const float* h_prev, Action action,
                            float r_total, float reward_ext, const float* s2,
                            bool done, const float homeo_targets[3], bool cheese,
                            bool wall_hit, uint64_t sim_steps) {
  ++lifetime_steps_;

  // Defensive: reject non-finite observations (should never happen).
  bool finite = true;
  for (int i = 0; i < input_size_; ++i)
    if (!std::isfinite(s[i]) || !std::isfinite(s2[i])) finite = false;
  for (int i = 0; i < rnnSize(); ++i) finite &= std::isfinite(h_prev[i]);
  for (int i = 0; i < 3; ++i) finite &= std::isfinite(homeo_targets[i]);
  finite &= std::isfinite(r_total) && std::isfinite(reward_ext);
  finite &= static_cast<int>(action) < 4;
  if (!finite) {
    ++invalid_updates_;
    return;
  }

  replay_.push(s, s2, h_prev, static_cast<int>(action), r_total, reward_ext,
               done, homeo_targets);

  // Episodic memory: keep only significant transitions (bounded capacity,
  // lowest-significance replacement when full).
  const float significance =
      cheese ? 3.0f
             : (std::fabs(r_total) > 1.0f   ? 1.5f
                    : (last_novelty_ > 0.6f ? 1.0f
                                            : (last_pred_error_ > 0.3f ? 0.8f
                                                                       : 0.1f)));
  if (significance >= 0.3f) {
    EpisodicEntry e;
    e.s.assign(s, s + input_size_);
    e.s2.assign(s2, s2 + input_size_);
    e.h.assign(h_prev, h_prev + rnnSize());
    e.action = static_cast<int>(action);
    e.reward = r_total;
    e.reward_ext = reward_ext;
    e.flags = 0;
    if (cheese) e.flags |= kFlagCheese;
    if (wall_hit) e.flags |= kFlagWall;
    if (last_novelty_ > 0.6f) e.flags |= kFlagHighNovelty;
    if (last_pred_error_ > 0.3f) e.flags |= kFlagHighPredError;
    e.significance = significance;
    e.insert_step = sim_steps;
    episodic_.add(std::move(e));
  }

  // Online training at a bounded cadence.
  if (lifetime_steps_ % static_cast<uint64_t>(cfg_.train_interval_steps) == 0 &&
      replay_.size() >= static_cast<size_t>(cfg_.batch_size)) {
    auto& idx = sample_indices_;
    replay_.sampleIndices(rng_, static_cast<size_t>(cfg_.batch_size), &idx,
                          &tb_is_w_);
    // Importance-sampling correction: w = (raw_odds)^-beta, max-normalized.
    // beta == 0 leaves every weight at 1.0 (no correction; deterministic).
    {
      const double progress = std::min(1.0, double(lifetime_steps_) / std::max(1, cfg_.exploration_decay_steps));
      const double beta = cfg_.per_is_beta > 0 ? cfg_.per_is_beta + (1.0 - cfg_.per_is_beta) * progress : 0;
      double wmax = 0.0;
      for (float& w : tb_is_w_) {
        w = static_cast<float>(std::pow(static_cast<double>(w), -beta));
        if (w > wmax) wmax = w;
      }
      if (wmax > 0.0) {
        const float inv_w = static_cast<float>(1.0 / wmax);
        for (float& w : tb_is_w_) w *= inv_w;
      } else {
        std::fill(tb_is_w_.begin(), tb_is_w_.end(), 1.0f);
      }
    }
    train_samples_.clear();
    train_samples_.reserve(idx.size());
    for (size_t i : idx) {
      train_samples_.push_back({replay_.obs(i), replay_.next_obs(i),
                                replay_.h(i), replay_.targets(i),
                                replay_.action(i), replay_.reward(i),
                                replay_.extReward(i), replay_.done(i)});
    }
    trainBatchOn(train_samples_, &idx);
  }
  if (cfg_.sequence_train_interval > 0 && cfg_.bptt_chunk_len >= 2 &&
      lifetime_steps_ % static_cast<uint64_t>(cfg_.sequence_train_interval) == 0 &&
      replay_.size() >= static_cast<size_t>(cfg_.bptt_chunk_len)) {
    const size_t len = static_cast<size_t>(cfg_.bptt_chunk_len);
    trainSequenceBatch(static_cast<size_t>(rng_.uniformInt(0, static_cast<int>(replay_.size() - len))), len);
  }
  if (done) resetRecurrent();
}

bool Agent::trainBatchOn(const std::vector<Sample>& samples,
                         const std::vector<size_t>* replay_idx) {
  const size_t n = samples.size();
  if (n == 0) return true;
  // Snapshot the current (valid) parameters for NaN rollback.
  online_.getParams(tb_snapshot_.data());

  std::fill(tb_grad_.begin(), tb_grad_.end(), 0.0f);
  tb_td_.resize(n, 0.0f);

  std::vector<float>& grad = tb_grad_;
  std::vector<float>& gb = tb_gb_;
  std::vector<float>& h = tb_h_;
  std::vector<float>& q = tb_q_;
  std::vector<float>& pred = tb_pred_;
  std::vector<float>& h2 = tb_h2_;
  std::vector<float>& q2 = tb_q2_;
  std::vector<float>& h2t = tb_h2t_;
  std::vector<float>& q2t = tb_q2t_;
  std::vector<float>& grad_q = tb_grad_q_;
  std::vector<float>& grad_p = tb_grad_p_;

  const double gamma = cfg_.discount_factor;
  // N-step bootstrap depth (config-clamped to [1,8]; 1 == classic 1-step TD
  // target). gpow[k] = gamma^k for the fixed worst-case depth.
  const size_t nstep = static_cast<size_t>(
      std::min(8, std::max(1, cfg_.n_step_returns)));
  float gpow[9];
  gpow[0] = 1.0f;
  for (size_t k = 1; k < 9; ++k)
    gpow[k] = gpow[k - 1] * static_cast<float>(gamma);
  const bool use_is = replay_idx != nullptr && !tb_is_w_.empty() &&
                      cfg_.per_is_beta > 0.0;
  // Huber threshold for the TD loss (per-sample gradient of smooth-L1).
  const float delta = static_cast<float>(cfg_.td_huber_delta);
  float targets[predOutStatic()];

  for (size_t si = 0; si < n; ++si) {
    const Sample& smp = samples[si];
    online_.forward(smp.s, smp.h_prev, h.data(), q.data(), pred.data());

    // --- TD target ---
    float y;
    if (replay_idx != nullptr) {
      // Reconstruct the continuation with each network's own recurrent dynamics.
      const size_t i0 = (*replay_idx)[si];
      const size_t avail = std::min(nstep, replay_.size() - i0);
      std::copy(h.begin(), h.end(), burn_h_.begin());
      target_.forward(smp.s, smp.h_prev, burn_ht_.data(), nullptr, nullptr);
      float acc = 0.0f;
      size_t m = 0;
      bool terminal = false;
      for (size_t k = 0; k < avail; ++k) {
        if (k > 0) {
          online_.forward(replay_.obs(i0+k), burn_h_.data(), burn_next_.data(), nullptr, nullptr);
          target_.forward(replay_.obs(i0+k), burn_ht_.data(), burn_next_t_.data(), nullptr, nullptr);
          burn_h_.swap(burn_next_); burn_ht_.swap(burn_next_t_);
        }
        acc += gpow[k] * replay_.reward(i0+k);
        m = k + 1;
        terminal = replay_.done(i0+k);
        if (terminal || replay_.boundary(i0+k)) break;
      }
      y = acc;
      if (!terminal) {
        const float* next = replay_.next_obs(i0 + m - 1);
        online_.forward(next, burn_h_.data(), h2.data(), q2.data(), nullptr);
        const int best = greedyAction(next, q2.data());
        target_.forward(next, burn_ht_.data(), h2t.data(), q2t.data(), nullptr);
        y += gpow[m] * q2t[best];
      }
    } else if (smp.done) {
      y = smp.reward;
    } else {
      online_.forward(smp.s2, h.data(), h2.data(), q2.data(), nullptr);
      const int best = greedyAction(smp.s2, q2.data());
      target_.forward(smp.s, smp.h_prev, burn_ht_.data(), nullptr, nullptr);
      target_.forward(smp.s2, burn_ht_.data(), h2t.data(), q2t.data(), nullptr);
      y = smp.reward + static_cast<float>(gamma) * q2t[best];
    }

    // Policy gradient: Huber-smoothed TD error (smooth-L1; large delta
    // degrades to plain MSE). The replay priority keeps the raw |TD| so rare
    // high-error transitions (cheese) stay emphasized.
    for (int k = 0; k < policyOut(); ++k) grad_q[k] = 0.0f;
    const float err = q[smp.action] - y;
    grad_q[smp.action] = std::fabs(err) <= delta
                             ? err
                             : delta * (err > 0.0f ? 1.0f : -1.0f);
    tb_td_[si] = std::fabs(err);

    // Prediction gradient (MSE over the 16 world-model outputs, external
    // reward as the target), scaled by the configured loss weight so it never
    // dominates the policy objective.
    updatePredictionTargets(smp.s, smp.s2, smp.tgt, smp.reward_ext, targets);
    for (int k = 0; k < predOut(); ++k)
      grad_p[k] = (pred[k] - targets[k]) *
                  static_cast<float>(cfg_.prediction_loss_weight);

    // Importance-sampling weight: the whole sample's loss (policy + world
    // model) is scaled; weights are max-normalized in observeAndTrain.
    const float w = use_is ? tb_is_w_[si] : 1.0f;

    online_.backward(smp.s, smp.h_prev, h.data(), grad_q.data(),
                     grad_p.data(), gb.data());
    for (size_t i = 0; i < grad.size(); ++i) grad[i] += w * gb[i];
  }

  // Prioritized replay: feed the per-sample TD errors back so rare
  // high-error transitions (cheese) keep being sampled.
  if (replay_idx) {
    for (size_t si = 0; si < n; ++si)
      replay_.updatePriority((*replay_idx)[si], tb_td_[si]);
  }

  // Average weighted gradients by batch size, retaining the IS correction.
  const float inv = 1.0f / static_cast<float>(n);
  for (size_t i = 0; i < grad.size(); ++i) grad[i] *= inv;

  // Gradient clipping (global norm).
  double norm = 0.0;
  for (float g : grad) norm += static_cast<double>(g) * g;
  norm = std::sqrt(norm);
  if (norm > cfg_.gradient_clip_norm) {
    const float scale = static_cast<float>(cfg_.gradient_clip_norm / norm);
    for (float& g : grad) g *= scale;
  }

  // Adam step with the age-scaled learning rate.
  ++training_updates_;
  adamUpdate(grad.data());

  // NaN / inf protection: restore the last valid parameters on instability.
  if (!online_.allFinite() || !allMomentsFinite()) {
    online_.setParams(tb_snapshot_.data());
    std::fill(adam_m_.begin(), adam_m_.end(), 0.0f);
    std::fill(adam_v_.begin(), adam_v_.end(), 0.0f);
    ++invalid_updates_;
    return false;
  }
  online_.updateUtility(grad.data(), cfg_.plasticity_utility_decay);
  softUpdateTarget();
  return true;
}

bool Agent::allMomentsFinite() const {
  for (float x : adam_m_)
    if (!std::isfinite(x)) return false;
  for (float x : adam_v_)
    if (!std::isfinite(x)) return false;
  return true;
}

void Agent::adamUpdate(const float* grad) {
  const double lr =
      cfg_.learning_rate * development_.learningRateScale(lifetime_steps_);
  const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
  const double t = static_cast<double>(training_updates_);
  const double c1 = 1.0 - std::pow(b1, t);
  const double c2 = 1.0 - std::pow(b2, t);
  const size_t n = adam_m_.size();
  std::vector<float>& params = adam_params_;
  online_.getParams(params.data());
  for (size_t i = 0; i < n; ++i) {
    adam_m_[i] =
        static_cast<float>(b1) * adam_m_[i] + static_cast<float>(1.0 - b1) * grad[i];
    adam_v_[i] = static_cast<float>(b2) * adam_v_[i] +
                 static_cast<float>(1.0 - b2) * grad[i] * grad[i];
    const double mhat = adam_m_[i] / c1;
    const double vhat = adam_v_[i] / c2;
    params[i] -= static_cast<float>(lr * mhat / (std::sqrt(vhat) + eps));
  }
  online_.maskDense(params.data());
  online_.maskDense(adam_m_.data());
  online_.maskDense(adam_v_.data());
  online_.setParams(params.data());
}

void Agent::softUpdateTarget() {
  std::vector<float>& po = soft_po_;
  std::vector<float>& pt = soft_pt_;
  online_.getParams(po.data());
  target_.getParams(pt.data());
  const float tau = static_cast<float>(cfg_.target_update_tau);
  for (size_t i = 0; i < pt.size(); ++i)
    pt[i] = (1.0f - tau) * pt[i] + tau * po[i];
  online_.maskDense(pt.data());
  target_.setParams(pt.data());
}

bool Agent::consolidationAvailable() const {
  return !episodic_.empty() ||
         replay_.size() >= static_cast<size_t>(cfg_.batch_size);
}

void Agent::gatherSamples(size_t count, std::vector<Sample>* out,
                          bool prefer_episodic) {
  out->clear();
  out->reserve(count);
  const size_t n_ep = episodic_.size();
  const size_t n_rp = replay_.size();
  auto ep_sample = [&]() {
    const EpisodicEntry& e =
        episodic_[rng_.uniformInt(0, static_cast<int>(n_ep - 1))];
    // done == cheese flag: episodes in this environment end at cheese.
    out->push_back({e.s.data(), e.s2.data(), e.h.data(), nullptr, e.action,
                    e.reward, e.reward_ext, (e.flags & kFlagCheese) != 0});
  };
  for (size_t i = 0; i < count; ++i) {
    const bool use_ep = prefer_episodic && n_ep > 0 && (i % 2 == 0 || n_rp == 0);
    if (use_ep) {
      ep_sample();
    } else if (n_rp > 0) {
      const size_t j = rng_.uniformInt(0, static_cast<int>(n_rp - 1));
      out->push_back({replay_.obs(j), replay_.next_obs(j), replay_.h(j),
                      replay_.targets(j), replay_.action(j), replay_.reward(j),
                      replay_.extReward(j), replay_.done(j)});
    } else if (n_ep > 0) {
      ep_sample();
    } else {
      return;  // nothing available
    }
  }
}

int Agent::consolidationTrainOps(int max_ops) {
  if (max_ops <= 0 || !consolidationAvailable()) return 0;
  int executed = 0;
  for (int i = 0; i < max_ops; ++i) {
    cons_samples_.clear();
    gatherSamples(static_cast<size_t>(cfg_.batch_size), &cons_samples_,
                  /*prefer_episodic=*/true);
    if (cons_samples_.empty()) break;
    if (trainBatchOn(cons_samples_, nullptr)) ++executed;
  }
  consolidation_train_ops_total_ += executed;
  return executed;
}

int Agent::consolidationSequenceOps(int max_ops) {
  if (max_ops <= 0) return 0;
  const int chunk = std::min(cfg_.bptt_chunk_len, static_cast<int>(kSeqChunkLen));
  if (chunk < 2) return 0;  // disabled (bptt_chunk_len 0) or too short
  const size_t len = static_cast<size_t>(chunk);
  const size_t n = replay_.size();
  if (n < len) return 0;  // not enough consecutive entries yet
  int executed = 0;
  for (int i = 0; i < max_ops; ++i) {
    const size_t start =
        static_cast<size_t>(rng_.uniformInt(0, static_cast<int>(n - len)));
    if (trainSequenceBatch(start, len)) ++executed;
  }
  consolidation_train_ops_total_ += executed;
  return executed;
}

bool Agent::trainSequenceBatch(size_t start, size_t len) {
  const int rnn = rnnSize(), po = policyOut(), pr = predOut();
  const size_t np = online_.paramCount();
  if (len < 2 || len > kSeqChunkLen || start + len > replay_.size()) return false;

  for (size_t j = 0; j + 1 < len; ++j) {
    if (replay_.boundary(start + j)) { len = j + 1; break; }
  }
  std::fill(seq_grad_total_.begin(), seq_grad_total_.end(), 0.0f);
  online_.getParams(seq_snapshot_.data());

  const float gamma = static_cast<float>(cfg_.discount_factor);
  const float delta = static_cast<float>(cfg_.td_huber_delta);

  // Forward chain: anchor = the stored pre-decision recurrent state of the
  // first entry; from there the hidden state is recomputed through the chunk
  // so every downstream target uses the true continuation state (unlike the
  // truncated-BPTT(1) stale-state replay approximation).
  burnIn(start, seq_hchain_.data(), seq_target_chain_.data());
  for (size_t j = 0; j < len; ++j) {
    const size_t ij = start + j;
    online_.forward(replay_.obs(ij), seq_hchain_.data() + j * rnn,
                    seq_hchain_.data() + (j + 1) * rnn,
                    seq_q_.data() + j * po, seq_pred_.data() + j * pr);
    target_.forward(replay_.obs(ij), seq_target_chain_.data() + j * rnn,
                    seq_target_chain_.data() + (j + 1) * rnn, nullptr, nullptr);
    float y;
    if (replay_.done(ij)) {
      y = replay_.reward(ij);
    } else {
      // Double-DQN target evaluated at the computed continuation state.
      online_.forward(replay_.next_obs(ij),
                      seq_hchain_.data() + (j + 1) * rnn, seq_h2_.data(),
                      seq_q2_.data(), nullptr);
      const int best = greedyAction(replay_.next_obs(ij), seq_q2_.data());
      target_.forward(replay_.next_obs(ij),
                      seq_target_chain_.data() + (j + 1) * rnn, seq_h2t_.data(),
                      seq_q2t_.data(), nullptr);
      y = replay_.reward(ij) + gamma * seq_q2t_[best];
    }
    std::fill(seq_grad_q_.begin() + static_cast<long>(j) * po,
              seq_grad_q_.begin() + static_cast<long>(j + 1) * po, 0.0f);
    const float err = seq_q_[static_cast<size_t>(j) * po + replay_.action(ij)] - y;
    replay_.updatePriority(ij, std::fabs(err));
    seq_grad_q_[static_cast<size_t>(j) * po + replay_.action(ij)] =
        std::fabs(err) <= delta ? err : delta * (err > 0.0f ? 1.0f : -1.0f);
    updatePredictionTargets(replay_.obs(ij), replay_.next_obs(ij),
                            replay_.targets(ij), replay_.extReward(ij),
                            seq_t_.data());
    for (int k = 0; k < pr; ++k)
      seq_grad_pred_[static_cast<size_t>(j) * pr + k] =
          (seq_pred_[static_cast<size_t>(j) * pr + k] - seq_t_[k]) *
          static_cast<float>(cfg_.prediction_loss_weight);
  }

  // Backward through the chain (last step first): the gradient of the loss
  // w.r.t. each hidden state flows into the next-earlier step via the
  // grad_h_prev output (head gradients + future flow) of NeuralNet::backward
  // (see gru.h). One Adam step per chunk, guarded like trainBatchOn.
  std::fill(seq_hp_b_.begin(), seq_hp_b_.end(), 0.0f);  // future flow (empty)
  for (size_t j = len; j-- > 0;) {
    const size_t ij = start + j;
    online_.backward(replay_.obs(ij), seq_hchain_.data() + j * rnn,
                     seq_h2_.data(), seq_grad_q_.data() + static_cast<long>(j) * po,
                     seq_grad_pred_.data() + static_cast<long>(j) * pr,
                     seq_gb_a_.data(), seq_hp_a_.data(), seq_hp_b_.data());
    for (size_t i = 0; i < np; ++i) seq_grad_total_[i] += seq_gb_a_[i];
    std::swap(seq_hp_a_, seq_hp_b_);  // this step's dL/dh_prev is the next one's future flow
  }

  // Average, clip, one Adam step, soft target update, NaN guard.
  const float inv = 1.0f / static_cast<float>(len);
  for (size_t i = 0; i < np; ++i) seq_grad_total_[i] *= inv;
  double norm = 0.0;
  for (float g : seq_grad_total_) norm += static_cast<double>(g) * g;
  norm = std::sqrt(norm);
  if (norm > cfg_.gradient_clip_norm) {
    const float scale = static_cast<float>(cfg_.gradient_clip_norm / norm);
    for (float& g : seq_grad_total_) g *= scale;
  }

  ++training_updates_;
  adamUpdate(seq_grad_total_.data());
  if (!online_.allFinite() || !allMomentsFinite()) {
    online_.setParams(seq_snapshot_.data());
    std::fill(adam_m_.begin(), adam_m_.end(), 0.0f);
    std::fill(adam_v_.begin(), adam_v_.end(), 0.0f);
    ++invalid_updates_;
    return false;
  }
  online_.updateUtility(seq_grad_total_.data(), cfg_.plasticity_utility_decay);
  softUpdateTarget();
  return true;
}

void Agent::plasticityEvaluate(const std::vector<float>& eval_obs,
                               int eval_passes) {
  // Snapshot for rollback on instability.
  std::vector<float> snapshot(online_.paramCount());
  online_.getParams(snapshot.data());
  std::vector<uint8_t> mask_snap(online_.maskBytes());
  online_.getMask(mask_snap.data());

  const size_t weight_bits = online_.paramCount();  // = utility_.size(); mask
  // bits beyond the weight layout are never set, so iterating over the
  // parameter count covers every active connection and keeps util[] in range.
  const size_t max_active =
      static_cast<size_t>(cfg_.plasticity_max_active_fraction *
                          static_cast<double>(weight_bits));
  const std::vector<float>& util = online_.utility();

  auto is_active = [&](size_t i, const std::vector<uint8_t>& m) {
    return (m[i / 8] & (1u << (i % 8))) != 0;
  };

  // Candidate active connections, weakest first.
  std::vector<size_t> candidates;
  for (size_t i = 0; i < weight_bits; ++i) {
    if (is_active(i, mask_snap)) candidates.push_back(i);
  }
  std::sort(candidates.begin(), candidates.end(),
            [&](size_t a, size_t b) { return util[a] < util[b]; });

  // Median utility of active connections; prune only demonstrably weak ones.
  float median = 0.0f;
  if (!candidates.empty()) {
    std::vector<float> sorted;
    sorted.reserve(candidates.size());
    for (size_t i : candidates) sorted.push_back(util[i]);
    std::sort(sorted.begin(), sorted.end());
    median = sorted[sorted.size() / 2];
  }
  const float weak_threshold =
      median * static_cast<float>(cfg_.plasticity_dormant_threshold_fraction);

  // Prune: never exceed the per-consolidation cap nor the hard active cap.
  const size_t prune_cap =
      static_cast<size_t>(cfg_.plasticity_rewire_per_consolidation) * 4;
  size_t pruned = 0;
  for (size_t i : candidates) {
    if (pruned >= prune_cap) break;
    if (online_.activeCount() <= max_active && util[i] >= weak_threshold) break;
    online_.zeroParam(i);
    ++pruned;
  }

  // Reactivate a bounded number of dormant connections that were previously
  // useful (highest utility first) with small fresh weights.
  size_t rewired = 0;
  std::vector<size_t> dormant;
  for (size_t i = 0; i < weight_bits; ++i) {
    if (!is_active(i, mask_snap)) dormant.push_back(i);
  }
  std::sort(dormant.begin(), dormant.end(),
            [&](size_t a, size_t b) { return util[a] > util[b]; });
  const double pscale = development_.plasticityScale(lifetime_steps_);
  const int rewire_cap =
      std::max(0, static_cast<int>(cfg_.plasticity_rewire_per_consolidation * pscale));
  for (size_t i : dormant) {
    if (rewired >= static_cast<size_t>(rewire_cap)) break;
    if (online_.activeCount() >= max_active) break;
    online_.reactivateParam(i, rng_, 0.5f);
    ++rewired;
  }

  // Strengthen the most useful connections (mild, bounded).
  const size_t strengthen_n = std::max<size_t>(1, candidates.size() / 20);
  for (size_t k = 0; k < strengthen_n && k < candidates.size(); ++k) {
    const size_t i = candidates[candidates.size() - 1 - k];
    online_.strengthenParam(i, static_cast<float>(cfg_.plasticity_strengthen_factor));
  }

  // Bounded evaluation: forward passes on recent observations must stay
  // finite and bounded, otherwise roll back.
  bool ok = online_.allFinite();
  if (ok && !eval_obs.empty()) {
    std::vector<float> h(rnnSize(), 0.0f), h2(rnnSize()), q(policyOut()), pred(predOut());
    const size_t n =
        std::min<size_t>(static_cast<size_t>(eval_passes),
                         eval_obs.size() / static_cast<size_t>(inputSize()));
    for (size_t p = 0; p < n && ok; ++p) {
      online_.forward(eval_obs.data() + p * inputSize(), h.data(), h2.data(),
                      q.data(), pred.data());
      h = h2;
      for (float v : q)
        if (!std::isfinite(v) || std::fabs(v) > 100.0f) ok = false;
      for (float v : pred)
        if (!std::isfinite(v)) ok = false;
    }
  }

  if (!ok) {
    online_.setParams(snapshot.data());
    online_.setMask(mask_snap.data());
    ++structural_rejected_;
    return;
  }
  ++structural_accepted_;
  pruned_total_ += pruned;
  rewired_total_ += rewired;
}

// --- persistence ---

void Agent::exportState(AgentState& out) const {
  out = AgentState{};
  out.input = inputSize();
  out.rnn = rnnSize();
  out.policy_out = policyOut();
  out.pred_out = predOut();
  out.lr = static_cast<float>(cfg_.learning_rate);
  out.gamma = static_cast<float>(cfg_.discount_factor);
  out.tau = static_cast<float>(cfg_.target_update_tau);
  out.training_steps = training_updates_;
  out.lifetime_steps = lifetime_steps_;
  out.invalid_updates = invalid_updates_;
  out.explored_count = explored_count_;
  out.epsilon = static_cast<float>(epsilon_);
  out.seed = rng_.seed();
  out.rng_state = rng_.saveState();
  out.novelty_ema = novelty_ema_;
  out.pred_loss_ema = pred_loss_ema_;
  out.uncertainty = uncertainty_;
  out.consolidation_cycles = static_cast<uint32_t>(consolidation_cycles_);
  out.consolidation_train_ops_total = consolidation_train_ops_total_;
  out.structural_accepted = static_cast<uint32_t>(structural_accepted_);
  out.structural_rejected = static_cast<uint32_t>(structural_rejected_);
  out.pruned_total = pruned_total_;
  out.rewired_total = rewired_total_;
  out.online_params.resize(online_.paramCount());
  online_.getParams(out.online_params.data());
  out.target_params.resize(target_.paramCount());
  target_.getParams(out.target_params.data());
  out.adam_m = adam_m_;
  out.adam_v = adam_v_;
  out.masks.resize(online_.maskBytes());
  online_.getMask(out.masks.data());
  out.utility.resize(online_.utility().size());
  online_.getUtility(out.utility.data());
  out.episodic_capacity = episodic_.capacity();
  episodic_.serializeTo(out.episodic_floats, out.episodic_meta);
  novelty_.serializeTo(out.novelty_table);
  replay_.serializeTo(out.replay_floats, out.replay_meta);
}

bool Agent::importState(const AgentState& in) {
  const int expected = online_.paramCount();
  if (in.input != inputSize() || in.rnn != rnnSize() ||
      in.policy_out != policyOut() || in.pred_out != predOut()) {
    return false;
  }
  if (in.online_params.size() != static_cast<size_t>(expected) ||
      in.target_params.size() != static_cast<size_t>(expected) ||
      in.adam_m.size() != static_cast<size_t>(expected) ||
      in.adam_v.size() != static_cast<size_t>(expected) ||
      in.masks.size() != online_.maskBytes() ||
      in.utility.size() != static_cast<size_t>(expected)) {
    return false;
  }
  if (!in.allFinite()) return false;
  for (float v : in.adam_v) if (v < 0.0f) return false;
  ReplayBuffer restored_replay(inputSize(), rnnSize(), replay_.capacity(), cfg_.per_priority_alpha);
  EpisodicMemory restored_episodic(inputSize(), rnnSize(), episodic_.capacity());
  if (!restored_replay.restoreFrom(in.replay_floats, in.replay_meta) ||
      !restored_episodic.restoreFrom(in.episodic_floats, in.episodic_meta)) return false;
  online_.setParams(in.online_params.data());
  target_.setParams(in.target_params.data());
  target_.setMask(in.masks.data());
  adam_m_ = in.adam_m;
  adam_v_ = in.adam_v;
  online_.setMask(in.masks.data());
  online_.setUtility(in.utility.data());
  if (!online_.allFinite() || !target_.allFinite() || !allMomentsFinite()) {
    return false;
  }

  training_updates_ = in.training_steps;
  lifetime_steps_ = in.lifetime_steps;
  invalid_updates_ = in.invalid_updates;
  explored_count_ = in.explored_count;
  epsilon_ = clampf(in.epsilon, 0.0, 1.0);
  novelty_ema_ = clampf(in.novelty_ema, 0.0f, 1.0f);
  pred_loss_ema_ = clampf(in.pred_loss_ema, 0.0f, 1.0f);
  uncertainty_ = clampf(in.uncertainty, 0.0f, 1.0f);
  consolidation_cycles_ = in.consolidation_cycles;
  consolidation_train_ops_total_ = in.consolidation_train_ops_total;
  structural_accepted_ = in.structural_accepted;
  structural_rejected_ = in.structural_rejected;
  pruned_total_ = in.pruned_total;
  rewired_total_ = in.rewired_total;

  episodic_ = std::move(restored_episodic);
  replay_ = std::move(restored_replay);
  novelty_.restoreFrom(in.novelty_table);
  resetRecurrent();  // working memory is transient across sessions
  return true;
}

}  // namespace sir
