#pragma once
// The rat's complete neural system: input -> GRU (working memory) -> policy
// head (Q-values) and prediction head (world-model outputs). Connection
// weights carry bit masks for structural plasticity (dormant connections are
// zeroed) and per-connection utility traces (EMA of |gradient * weight|).

#include "learning/gru.h"
#include "utility/rng.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sir {

class NeuralNet {
 public:
  // policy_out = number of actions (4); pred_out = prediction head size (16).
  NeuralNet(int input, int rnn_hidden, int policy_out, int pred_out, Rng& rng);

  // q = Wq h + bq ; pred = Wp h + bp ; h = GRU(x, h_prev).
  void forward(const float* x, const float* h_prev, float* h_out, float* q,
               float* pred) const;

  // Backward for one step. grad_q is a policy_out vector (caller fills only
  // the taken action with dL/dq); grad_pred is pred_out. Fills grad_p
  // (paramCount floats). Recomputed forward internally; `h_out` is a scratch
  // buffer (the recomputed hidden state is written into it).
  void backward(const float* x, const float* h_prev, float* h_out,
                const float* grad_q, const float* grad_pred,
                float* grad_p) const;

  int input() const { return in_; }
  int rnnHidden() const { return rnn_; }
  int policyOut() const { return po_; }
  int predOut() const { return pr_; }

  // Total parameter count: gru + Wq + bq + Wp + bp.
  static int paramCount(int input, int rnn, int po, int pr);
  int paramCount() const { return paramCount(in_, rnn_, po_, pr_); }

  void getParams(float* out) const;
  void setParams(const float* p);
  bool allFinite() const;

  // --- structural plasticity ---
  // Masks cover the connection weights only (biases stay dense). One bit per
  // parameter in the same layout as getParams; bit set = active.
  size_t maskBytes() const { return (paramCount() + 7) / 8; }
  void getMask(uint8_t* out) const;
  void setMask(const uint8_t* in);
  size_t activeCount() const;
  size_t dormantCount() const { return paramCount() - activeCount(); }

  // Per-parameter init scale (used when reactivating dormant connections).
  const std::vector<float>& initScales() const { return init_scale_; }

  // Utility trace: u = decay*u + (1-decay)*|grad*param| (active params only).
  void updateUtility(const float* grad, double decay);
  const std::vector<float>& utility() const { return utility_; }
  void getUtility(float* out) const;
  void setUtility(const float* in);

  // Plasticity primitives (used only during consolidation).
  void zeroParam(size_t i);  // weight = 0, mask off
  void reactivateParam(size_t i, Rng& rng, float scale);
  void strengthenParam(size_t i, float factor);  // weight *= (1 + factor)

 private:
  void computeInitScales();

  // Weight-index (mask order) <-> storage helpers. `weightPtr` returns the
  // underlying weight storage for a mask/utility weight index, or nullptr
  // for biases / out of range.
  float* weightPtr(size_t weight_index);
  const float* weightPtr(size_t weight_index) const;
  void writeWeight(size_t weight_index, float v);

  int in_;
  int rnn_;
  int po_;
  int pr_;
  GruLayer gru_;
  std::vector<float> wq_, bq_, wp_, bp_;
  std::vector<uint8_t> mask_;  // bit-packed, one bit per param (weights only)
  std::vector<float> utility_;
  std::vector<float> init_scale_;  // per param (weights only)
};

}  // namespace sir
