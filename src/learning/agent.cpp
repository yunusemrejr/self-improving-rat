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
              static_cast<size_t>(cfg.replay_capacity)),
      episodic_(kObservationBase * cfg.observation_frames, cfg.rnn_hidden,
                static_cast<size_t>(cfg.episodic_memory_capacity)),
      novelty_(static_cast<size_t>(cfg.novelty_table_capacity)),
      h_(cfg.rnn_hidden, 0.0f),
      last_pred_(predOutStatic(), 0.0f),
      input_size_(kObservationBase * cfg.observation_frames) {
  const size_t n = online_.paramCount();
  adam_m_.assign(n, 0.0f);
  adam_v_.assign(n, 0.0f);
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

void Agent::resetRecurrent() { std::fill(h_.begin(), h_.end(), 0.0f); }

Agent::Decision Agent::selectAction(const float* obs) {
  epsilon_ = computeEpsilon(lifetime_steps_);
  std::vector<float> h2(rnnSize()), q(policyOut());
  // h_out must not alias h_prev (see GruLayer).
  online_.forward(obs, h_.data(), h2.data(), q.data(), last_pred_.data());
  h_ = h2;

  if (rng_.uniform01() < epsilon_) {
    ++explored_count_;
    return {static_cast<Action>(rng_.uniformInt(0, 3)), true};
  }
  int best = 0;
  for (int k = 1; k < policyOut(); ++k)
    if (q[k] > q[best]) best = k;
  return {static_cast<Action>(best), false};
}

void Agent::updatePredictionTargets(const float* s, const float* s2,
                                    const float* homeo_targets, float reward_ext,
                                    float* out) const {
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
    std::vector<size_t> idx;
    replay_.sampleIndices(rng_, static_cast<size_t>(cfg_.batch_size), &idx);
    std::vector<Sample> samples;
    samples.reserve(idx.size());
    for (size_t i : idx) {
      samples.push_back({replay_.obs(i), replay_.next_obs(i), replay_.h(i),
                         replay_.targets(i), replay_.action(i), replay_.reward(i),
                         replay_.extReward(i), replay_.done(i)});
    }
    trainBatchOn(samples, &idx);
  }
}

bool Agent::trainBatchOn(const std::vector<Sample>& samples,
                         const std::vector<size_t>* replay_idx) {
  // Snapshot the current (valid) parameters for NaN rollback.
  std::vector<float> snapshot(online_.paramCount());
  online_.getParams(snapshot.data());

  std::vector<float> grad(online_.paramCount(), 0.0f);
  std::vector<float> gb(online_.paramCount());
  std::vector<float> h(rnnSize()), q(policyOut()), pred(predOut());
  std::vector<float> h2(rnnSize()), q2(policyOut()), h2t(rnnSize()), q2t(policyOut());
  std::vector<float> grad_q(policyOut()), grad_p(predOut());
  std::vector<float> td(samples.size(), 0.0f);
  float targets[16];

  for (size_t si = 0; si < samples.size(); ++si) {
    const Sample& smp = samples[si];
    online_.forward(smp.s, smp.h_prev, h.data(), q.data(), pred.data());

    // Double-DQN target: a* = argmax online Q(s2, h_prev), value from the
    // target network (recurrent state stays the stored h_prev: truncated
    // BPTT with stale recurrent states, documented approximation).
    float y;
    if (smp.done) {
      y = smp.reward;
    } else {
      online_.forward(smp.s2, smp.h_prev, h2.data(), q2.data(), nullptr);
      int best = 0;
      for (int k = 1; k < policyOut(); ++k)
        if (q2[k] > q2[best]) best = k;
      target_.forward(smp.s2, smp.h_prev, h2t.data(), q2t.data(), nullptr);
      y = smp.reward +
          static_cast<float>(cfg_.discount_factor) * q2t[best];
    }

    // Policy gradient (L = 0.5 * (q[a] - y)^2).
    for (int k = 0; k < policyOut(); ++k) grad_q[k] = 0.0f;
    grad_q[smp.action] = q[smp.action] - y;
    td[si] = std::fabs(q[smp.action] - y);

    // Prediction gradient (MSE over the 16 world-model outputs), scaled by
    // the configured prediction-loss weight so it never dominates the policy
    // objective. The target uses the external reward (what the model is asked
    // to predict), not the composite RL reward.
    updatePredictionTargets(smp.s, smp.s2, smp.tgt, smp.reward_ext, targets);
    for (int k = 0; k < predOut(); ++k)
      grad_p[k] = (pred[k] - targets[k]) *
                  static_cast<float>(cfg_.prediction_loss_weight);

    online_.backward(smp.s, smp.h_prev, h.data(), grad_q.data(), grad_p.data(),
                     gb.data());
    for (size_t i = 0; i < grad.size(); ++i) grad[i] += gb[i];
  }

  // Prioritized replay: feed the per-sample TD errors back so rare
  // high-error transitions (cheese) keep being sampled.
  if (replay_idx) {
    for (size_t si = 0; si < samples.size(); ++si)
      replay_.updatePriority((*replay_idx)[si], td[si]);
  }

  // Average the accumulated gradients.
  const float inv = 1.0f / static_cast<float>(samples.size());
  for (size_t i = 0; i < grad.size(); ++i) grad[i] *= inv;

  // Gradient clipping (global norm).
  double norm = 0.0;
  for (float g : grad) norm += static_cast<double>(g) * g;
  norm = std::sqrt(norm);
  if (norm > cfg_.gradient_clip_norm) {
    const float scale = static_cast<float>(cfg_.gradient_clip_norm / norm);
    for (float& g : grad) g *= scale;
  }

  // Utility trace before the parameter update (structural plasticity).
  online_.updateUtility(grad.data(), cfg_.plasticity_utility_decay);

  // Adam step with the age-scaled learning rate.
  ++training_updates_;
  adamUpdate(grad.data());
  softUpdateTarget();

  // NaN / inf protection: restore the last valid parameters on instability.
  if (!online_.allFinite() || !allMomentsFinite()) {
    online_.setParams(snapshot.data());
    std::fill(adam_m_.begin(), adam_m_.end(), 0.0f);
    std::fill(adam_v_.begin(), adam_v_.end(), 0.0f);
    ++invalid_updates_;
    return false;
  }
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
  std::vector<float> params(online_.paramCount());
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
  online_.setParams(params.data());
}

void Agent::softUpdateTarget() {
  std::vector<float> po(online_.paramCount()), pt(online_.paramCount());
  online_.getParams(po.data());
  target_.getParams(pt.data());
  const float tau = static_cast<float>(cfg_.target_update_tau);
  for (size_t i = 0; i < pt.size(); ++i)
    pt[i] = (1.0f - tau) * pt[i] + tau * po[i];
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
    std::vector<Sample> samples;
    gatherSamples(static_cast<size_t>(cfg_.batch_size), &samples,
                  /*prefer_episodic=*/true);
    if (samples.empty()) break;
    if (trainBatchOn(samples, nullptr)) ++executed;
  }
  consolidation_train_ops_total_ += executed;
  return executed;
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
  online_.setParams(in.online_params.data());
  target_.setParams(in.target_params.data());
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

  episodic_.clear();
  if (!in.episodic_floats.empty() || !in.episodic_meta.empty()) {
    if (!episodic_.restoreFrom(in.episodic_floats, in.episodic_meta)) return false;
  }
  novelty_.restoreFrom(in.novelty_table);
  resetRecurrent();  // working memory is transient across sessions
  return true;
}

}  // namespace sir
