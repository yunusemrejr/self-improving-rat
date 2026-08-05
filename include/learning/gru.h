#pragma once
// Compact gated recurrent unit (single timestep). Used as the rat's working
// memory: h_t = GRU(x_t, h_{t-1}). Training uses truncated BPTT of length 1:
// h_{t-1} is treated as a constant input (its stored value is part of the
// replay transition), so the recurrent state never grows and gradients stay
// bounded.

#include "utility/rng.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sir {

class GruLayer {
 public:
  GruLayer(int input, int hidden, Rng& rng);

  int input() const { return in_; }
  int hidden() const { return h_; }
  static int paramCount(int input, int hidden);

  // h_out = GRU(x, h_prev). h_out and h_prev may not alias.
  void forward(const float* x, const float* h_prev, float* h_out) const;

  // Backward for one step (h_prev treated as constant). Fills grad_p
  // (paramCount floats) with the gradient w.r.t. the parameters, given
  // dL/dh (hidden floats). Forward values are recomputed inside.
  void backward(const float* x, const float* h_prev, const float* h_out,
                const float* grad_h, float* grad_p) const;

  // Parameter layout: [Wz (in*h) | Uz (h*h) | bz (h) | Wr | Ur | br | Wh | Uh | bh],
  // row-major: Wg[i*h + j] = weight from input i to gate j; Ug[j*h + k] =
  // weight from recurrent unit j to gate k.
  void getParams(float* out) const;
  void setParams(const float* p);

  // Pointer to a weight at a dense parameter offset (weights only, not
  // biases), or nullptr if the offset points at a bias. Used by the
  // structural-plasticity code in NeuralNet.
  float* denseWeightPtr(size_t dense_offset);
  const float* denseWeightPtr(size_t dense_offset) const;

  bool allFinite() const;

 private:
  int in_;
  int h_;
  std::vector<float> wz_, uz_, bz_, wr_, ur_, br_, wh_, uh_, bh_;
};

}  // namespace sir
