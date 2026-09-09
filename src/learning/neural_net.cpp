#include "learning/neural_net.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sir {

namespace {

// Parameter layout metadata.
// Dense (params) layout: [GRU dense | Wq | bq | Wp | bp], where the GRU dense
// layout is [Wz | Uz | bz | Wr | Ur | br | Wh | Uh | bh].
// Mask/utility layout (weights only, in mask order):
//   [Wz | Uz | Wr | Ur | Wh | Uh | Wq | Wp]  (same sizes as dense, biases
//   excluded). Mask bit i corresponds to weight-index i in this order.
struct Layout {
  size_t w_z, u_z, w_r, u_r, w_h, u_h, w_q, w_p;  // per-block sizes
  size_t weights_total;
};

Layout layoutOf(int in, int rnn, int po, int pr) {
  const size_t n_wi = static_cast<size_t>(in) * rnn;
  const size_t n_wr = static_cast<size_t>(rnn) * rnn;
  Layout L{n_wi, n_wr, n_wi, n_wr, n_wi, n_wr,
           static_cast<size_t>(rnn) * po, static_cast<size_t>(rnn) * pr, 0};
  L.weights_total =
      L.w_z + L.u_z + L.w_r + L.u_r + L.w_h + L.u_h + L.w_q + L.w_p;
  return L;
}

// Maps a mask/utility weight-index to its dense parameter offset (or
// SIZE_MAX if out of range). Verified against the dense layout ordering.
size_t weightToDense(size_t i, int rnn, int po, const Layout& L) {
  if (i < L.w_z) return i;
  if (i < L.w_z + L.u_z) return i;
  if (i < L.w_z + L.u_z + L.w_r) return i + static_cast<size_t>(rnn);
  if (i < L.w_z + L.u_z + L.w_r + L.u_r) return i + static_cast<size_t>(rnn);
  if (i < L.w_z + L.u_z + L.w_r + L.u_r + L.w_h)
    return i + 2 * static_cast<size_t>(rnn);
  if (i < L.w_z + L.u_z + L.w_r + L.u_r + L.w_h + L.u_h)
    return i + 2 * static_cast<size_t>(rnn);
  if (i < L.w_z + L.u_z + L.w_r + L.u_r + L.w_h + L.u_h + L.w_q)
    return i + 3 * static_cast<size_t>(rnn);
  if (i < L.weights_total) return i + 3 * static_cast<size_t>(rnn) + po;
  return SIZE_MAX;
}

}  // namespace

NeuralNet::NeuralNet(int input, int rnn_hidden, int policy_out, int pred_out,
                     Rng& rng)
    : in_(input),
      rnn_(rnn_hidden),
      po_(policy_out),
      pr_(pred_out),
      gru_(input, rnn_hidden, rng) {
  if (input < 1 || rnn_hidden < 1 || policy_out < 1 || pred_out < 1)
    throw std::runtime_error("neural net dimensions");
  const double s_q = std::sqrt(6.0 / (rnn_hidden + policy_out));
  const double s_p = std::sqrt(6.0 / (rnn_hidden + pred_out));
  wq_.assign(static_cast<size_t>(rnn_hidden) * policy_out, 0.0f);
  wp_.assign(static_cast<size_t>(rnn_hidden) * pred_out, 0.0f);
  bq_.assign(policy_out, 0.0f);
  bp_.assign(pred_out, 0.0f);
  for (auto& x : wq_) x = static_cast<float>(rng.uniform(-s_q, s_q));
  for (auto& x : wp_) x = static_cast<float>(rng.uniform(-s_p, s_p));
  // Masks: one bit per parameter, but only weight bits are ever set (biases
  // stay dense and are not counted as connections).
  mask_.assign(maskBytes(), 0x00);
  const Layout L = layoutOf(in_, rnn_, po_, pr_);
  for (size_t i = 0; i < L.weights_total; ++i)
    mask_[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
  computeInitScales();
  utility_.assign(paramCount(), 0.0f);
}

int NeuralNet::paramCount(int input, int rnn, int po, int pr) {
  return GruLayer::paramCount(input, rnn) + rnn * po + po + rnn * pr + pr;
}

void NeuralNet::computeInitScales() {
  const Layout L = layoutOf(in_, rnn_, po_, pr_);
  init_scale_.assign(L.weights_total, 0.0f);
  const float s_in = static_cast<float>(std::sqrt(6.0 / (in_ + rnn_)));
  const float s_rec = static_cast<float>(std::sqrt(6.0 / (rnn_ + rnn_)));
  const float s_q = static_cast<float>(std::sqrt(6.0 / (rnn_ + po_)));
  const float s_p = static_cast<float>(std::sqrt(6.0 / (rnn_ + pr_)));
  size_t o = 0;
  for (size_t i = 0; i < L.w_z; ++i) init_scale_[o + i] = s_in;
  o += L.w_z;
  for (size_t i = 0; i < L.u_z; ++i) init_scale_[o + i] = s_rec;
  o += L.u_z;
  for (size_t i = 0; i < L.w_r; ++i) init_scale_[o + i] = s_in;
  o += L.w_r;
  for (size_t i = 0; i < L.u_r; ++i) init_scale_[o + i] = s_rec;
  o += L.u_r;
  for (size_t i = 0; i < L.w_h; ++i) init_scale_[o + i] = s_in;
  o += L.w_h;
  for (size_t i = 0; i < L.u_h; ++i) init_scale_[o + i] = s_rec;
  o += L.u_h;
  for (size_t i = 0; i < L.w_q; ++i) init_scale_[o + i] = s_q;
  o += L.w_q;
  for (size_t i = 0; i < L.w_p; ++i) init_scale_[o + i] = s_p;
}

void NeuralNet::getParams(float* out) const {
  gru_.getParams(out);
  size_t o = GruLayer::paramCount(in_, rnn_);
  std::copy(wq_.begin(), wq_.end(), out + o);
  o += wq_.size();
  std::copy(bq_.begin(), bq_.end(), out + o);
  o += bq_.size();
  std::copy(wp_.begin(), wp_.end(), out + o);
  o += wp_.size();
  std::copy(bp_.begin(), bp_.end(), out + o);
}

void NeuralNet::setParams(const float* p) {
  gru_.setParams(p);
  size_t o = GruLayer::paramCount(in_, rnn_);
  std::copy(p + o, p + o + wq_.size(), wq_.begin());
  o += wq_.size();
  std::copy(p + o, p + o + bq_.size(), bq_.begin());
  o += bq_.size();
  std::copy(p + o, p + o + wp_.size(), wp_.begin());
  o += wp_.size();
  std::copy(p + o, p + o + bp_.size(), bp_.begin());
}

bool NeuralNet::allFinite() const {
  if (!gru_.allFinite()) return false;
  for (const auto* v : {&wq_, &bq_, &wp_, &bp_})
    for (float x : *v)
      if (!std::isfinite(x)) return false;
  return true;
}

void NeuralNet::forward(const float* x, const float* h_prev, float* h_out,
                        float* q, float* pred) const {
  gru_.forward(x, h_prev, h_out);
  if (q) for (int k = 0; k < po_; ++k) {
    float s = bq_[k];
    for (int j = 0; j < rnn_; ++j)
      s += wq_[static_cast<size_t>(j) * po_ + k] * h_out[j];
    q[k] = s;
  }
  if (!pred) return;  // prediction head not required (e.g. target evaluation)
  for (int k = 0; k < pr_; ++k) {
    float s = bp_[k];
    for (int j = 0; j < rnn_; ++j)
      s += wp_[static_cast<size_t>(j) * pr_ + k] * h_out[j];
    // Prediction outputs are sigmoid-bounded to [0,1].
    pred[k] = 1.0f / (1.0f + std::exp(-s));
  }
}

void NeuralNet::backward(const float* x, const float* h_prev, float* h_out,
                         const float* grad_q, const float* grad_pred,
                         float* grad_p, float* grad_h_prev,
                         const float* grad_h_extra) const {
  // Recompute the forward pass (deterministic) including predictions so the
  // sigmoid chain rule can be applied exactly.
  scratch_q_.resize(po_);
  scratch_pred_.resize(pr_);
  scratch_dp_.resize(pr_);
  scratch_grad_h_.assign(rnn_, 0.0f);
  std::vector<float>& q = scratch_q_;
  std::vector<float>& pred = scratch_pred_;
  forward(x, h_prev, h_out, q.data(), pred.data());

  // Prediction head: grad_pred is dL/d(pred_output); pre-activation gradient
  // is grad_pred * pred * (1 - pred).
  std::vector<float>& dp = scratch_dp_;
  for (int k = 0; k < pr_; ++k) dp[k] = grad_pred[k] * pred[k] * (1.0f - pred[k]);

  // Gradient w.r.t. the hidden state from both heads (plus an optional
  // future-step flow when unrolling multiple steps; see grad_h_extra).
  std::vector<float>& grad_h = scratch_grad_h_;
  for (int j = 0; j < rnn_; ++j) {
    for (int k = 0; k < po_; ++k)
      grad_h[j] += wq_[static_cast<size_t>(j) * po_ + k] * grad_q[k];
    for (int k = 0; k < pr_; ++k)
      grad_h[j] += wp_[static_cast<size_t>(j) * pr_ + k] * dp[k];
    if (grad_h_extra) grad_h[j] += grad_h_extra[j];
  }

  // Head parameter gradients (dense layout, after the GRU block).
  const size_t n_gru = GruLayer::paramCount(in_, rnn_);
  std::fill(grad_p, grad_p + paramCount(), 0.0f);
  size_t o = n_gru;
  for (int j = 0; j < rnn_; ++j)
    for (int k = 0; k < po_; ++k)
      grad_p[o + static_cast<size_t>(j) * po_ + k] = grad_q[k] * h_out[j];
  o += static_cast<size_t>(rnn_) * po_;
  std::copy(grad_q, grad_q + po_, grad_p + o);
  o += po_;
  for (int j = 0; j < rnn_; ++j)
    for (int k = 0; k < pr_; ++k)
      grad_p[o + static_cast<size_t>(j) * pr_ + k] = dp[k] * h_out[j];
  o += static_cast<size_t>(rnn_) * pr_;
  std::copy(dp.begin(), dp.end(), grad_p + o);

  // GRU parameter gradients. Single-step callers (grad_h_prev == nullptr)
  // keep the documented truncated-BPTT(1) semantics; unrolled callers
  // receive the recurrent gradient w.r.t. h_prev to continue the chain.
  gru_.backward(x, h_prev, h_out, grad_h.data(), grad_p, grad_h_prev);
  maskDense(grad_p);
}

void NeuralNet::maskDense(float* dense) const {
  const Layout layout = layoutOf(in_, rnn_, po_, pr_);
  for (size_t i = 0; i < layout.weights_total; ++i)
    if (!(mask_[i / 8] & (1u << (i % 8))))
      dense[weightToDense(i, rnn_, po_, layout)] = 0.0f;
}

void NeuralNet::getMask(uint8_t* out) const {
  std::copy(mask_.begin(), mask_.end(), out);
}

void NeuralNet::setMask(const uint8_t* in) {
  std::copy(in, in + mask_.size(), mask_.begin());
}

size_t NeuralNet::activeCount() const {
  size_t n = 0;
  for (size_t i = 0; i < mask_.size(); ++i) {
    for (int b = 0; b < 8; ++b)
      if (mask_[i] & (1u << b)) ++n;
  }
  return n;
}

void NeuralNet::updateUtility(const float* grad, double decay) {
  const Layout L = layoutOf(in_, rnn_, po_, pr_);
  const size_t n = paramCount();
  // The mask/utility layout uses weight indices (biases excluded), while the
  // gradient buffer uses the dense parameter layout (biases interleaved).
  // Walk the weight-index space and fetch each weight's dense gradient via
  // weightToDense; never interpret a bias offset as a weight index.
  for (size_t i = 0; i < L.weights_total; ++i) {
    const bool active = (mask_[i / 8] & (1u << (i % 8))) != 0;
    if (!active) continue;
    const size_t off = weightToDense(i, rnn_, po_, L);
    if (off == SIZE_MAX || off >= n) continue;  // defensive
    const float* w = weightPtr(i);
    if (!w) continue;  // defensive; cannot occur for weight indices
    utility_[i] = static_cast<float>(decay) * utility_[i] +
                  static_cast<float>(1.0 - decay) * std::fabs(grad[off]) *
                      std::fabs(*w);
  }
}

void NeuralNet::getUtility(float* out) const {
  std::copy(utility_.begin(), utility_.end(), out);
}

void NeuralNet::setUtility(const float* in) {
  std::copy(in, in + utility_.size(), utility_.begin());
}

void NeuralNet::zeroParam(size_t i) {
  const Layout L = layoutOf(in_, rnn_, po_, pr_);
  if (i >= L.weights_total) return;
  mask_[i / 8] &= static_cast<uint8_t>(~(1u << (i % 8)));
  writeWeight(i, 0.0f);
  utility_[i] = 0.0f;
}

void NeuralNet::reactivateParam(size_t i, Rng& rng, float scale) {
  const Layout L = layoutOf(in_, rnn_, po_, pr_);
  if (i >= L.weights_total) return;
  mask_[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
  const float s = init_scale_[i] * scale;
  writeWeight(i, static_cast<float>(rng.uniform(-s, s)));
}

void NeuralNet::strengthenParam(size_t i, float factor) {
  const Layout L = layoutOf(in_, rnn_, po_, pr_);
  if (i >= L.weights_total) return;
  float* w = weightPtr(i);
  if (w) *w *= (1.0f + factor);
}

// --- private helpers ---

float* NeuralNet::weightPtr(size_t weight_index) {
  const Layout L = layoutOf(in_, rnn_, po_, pr_);
  const size_t off = weightToDense(weight_index, rnn_, po_, L);
  if (off == SIZE_MAX) return nullptr;
  const size_t n_gru = GruLayer::paramCount(in_, rnn_);
  if (off < n_gru) return gru_.denseWeightPtr(off);
  if (off < n_gru + wq_.size()) return wq_.data() + (off - n_gru);
  if (off < n_gru + wq_.size() + bq_.size()) return nullptr;  // bias
  if (off < n_gru + wq_.size() + bq_.size() + wp_.size())
    return wp_.data() + (off - n_gru - wq_.size() - bq_.size());
  return nullptr;
}

const float* NeuralNet::weightPtr(size_t weight_index) const {
  return const_cast<NeuralNet*>(this)->weightPtr(weight_index);
}

void NeuralNet::writeWeight(size_t weight_index, float v) {
  float* w = weightPtr(weight_index);
  if (w) *w = v;
}

}  // namespace sir
