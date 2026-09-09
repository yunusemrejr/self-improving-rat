#include "learning/gru.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sir {

namespace {
inline float sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }
}  // namespace

GruLayer::GruLayer(int input, int hidden, Rng& rng) : in_(input), h_(hidden) {
  if (input < 1 || hidden < 1) throw std::runtime_error("gru dimensions");
  // Glorot-style init per gate matrix (input and recurrent parts).
  const double s_in = std::sqrt(6.0 / (input + hidden));
  const double s_rec = std::sqrt(6.0 / (hidden + hidden));
  auto fill = [&](std::vector<float>& v, size_t n, double scale) {
    v.resize(n);
    for (auto& x : v) x = static_cast<float>(rng.uniform(-scale, scale));
  };
  fill(wz_, static_cast<size_t>(input) * hidden, s_in);
  fill(uz_, static_cast<size_t>(hidden) * hidden, s_rec);
  fill(bz_, hidden, 0.0);
  fill(wr_, static_cast<size_t>(input) * hidden, s_in);
  fill(ur_, static_cast<size_t>(hidden) * hidden, s_rec);
  fill(br_, hidden, 0.0);
  fill(wh_, static_cast<size_t>(input) * hidden, s_in);
  fill(uh_, static_cast<size_t>(hidden) * hidden, s_rec);
  fill(bh_, hidden, 0.0);
}

int GruLayer::paramCount(int input, int hidden) {
  return 3 * (input * hidden + hidden * hidden + hidden);
}

void GruLayer::getParams(float* out) const {
  size_t o = 0;
  for (const auto* v : {&wz_, &uz_, &bz_, &wr_, &ur_, &br_, &wh_, &uh_, &bh_}) {
    std::copy(v->begin(), v->end(), out + o);
    o += v->size();
  }
}

void GruLayer::setParams(const float* p) {
  size_t o = 0;
  for (auto* v : {&wz_, &uz_, &bz_, &wr_, &ur_, &br_, &wh_, &uh_, &bh_}) {
    std::copy(p + o, p + o + v->size(), v->begin());
    o += v->size();
  }
}

bool GruLayer::allFinite() const {
  for (const auto* v :
       {&wz_, &uz_, &bz_, &wr_, &ur_, &br_, &wh_, &uh_, &bh_}) {
    for (float x : *v)
      if (!std::isfinite(x)) return false;
  }
  return true;
}

float* GruLayer::denseWeightPtr(size_t dense_offset) {
  const size_t n_wi = static_cast<size_t>(in_) * h_;
  const size_t n_wr = static_cast<size_t>(h_) * h_;
  size_t o = 0;
  // Dense layout: [Wz | Uz | bz | Wr | Ur | br | Wh | Uh | bh].
  if (dense_offset < o + n_wi) return wz_.data() + dense_offset;
  o += n_wi;
  if (dense_offset < o + n_wr) return uz_.data() + (dense_offset - o);
  o += n_wr + h_;  // skip bz
  if (dense_offset < o + n_wi) return wr_.data() + (dense_offset - o);
  o += n_wi;
  if (dense_offset < o + n_wr) return ur_.data() + (dense_offset - o);
  o += n_wr + h_;  // skip br
  if (dense_offset < o + n_wi) return wh_.data() + (dense_offset - o);
  o += n_wi;
  if (dense_offset < o + n_wr) return uh_.data() + (dense_offset - o);
  return nullptr;  // bias or out of range
}

const float* GruLayer::denseWeightPtr(size_t dense_offset) const {
  return const_cast<GruLayer*>(this)->denseWeightPtr(dense_offset);
}

void GruLayer::ensureScratch() const {
  const int h = h_;
  fz_.resize(h);
  fr_.resize(h);
  fc_.resize(h);
  bz2_.resize(h);
  br2_.resize(h);
  bc2_.resize(h);
  baz_.resize(h);
  bar_.resize(h);
  bac_.resize(h);
  bdc_.resize(h);
  bdz_.resize(h);
  bda_.resize(h);
  bdr_.resize(h);
  bdlz_.resize(h);
  bdlr_.resize(h);
}

void GruLayer::forward(const float* x, const float* h_prev, float* h_out) const {
  ensureScratch();
  const int in = in_, h = h_;
  std::vector<float>& z = fz_;
  std::vector<float>& r = fr_;
  std::vector<float>& c = fc_;
  // z = sigmoid(Wz x + Uz h_prev + bz)
  // r = sigmoid(Wr x + Ur h_prev + br)
  // c = tanh(Wh x + Uh (r * h_prev) + bh)
  // h = (1 - z) * h_prev + z * c
  for (int j = 0; j < h; ++j) {
    float az = bz_[j], ar = br_[j], ac = bh_[j];
    for (int i = 0; i < in; ++i) {
      az += wz_[static_cast<size_t>(i) * h + j] * x[i];
      ar += wr_[static_cast<size_t>(i) * h + j] * x[i];
      ac += wh_[static_cast<size_t>(i) * h + j] * x[i];
    }
    for (int k = 0; k < h; ++k) {
      az += uz_[static_cast<size_t>(k) * h + j] * h_prev[k];
      ar += ur_[static_cast<size_t>(k) * h + j] * h_prev[k];
    }
    z[j] = sigmoid(az);
    r[j] = sigmoid(ar);
  }
  for (int j = 0; j < h; ++j) {
    float ac = bh_[j];
    for (int i = 0; i < in; ++i) {
      ac += wh_[static_cast<size_t>(i) * h + j] * x[i];
    }
    for (int k = 0; k < h; ++k) {
      ac += uh_[static_cast<size_t>(k) * h + j] * (r[k] * h_prev[k]);
    }
    c[j] = std::tanh(ac);
    h_out[j] = (1.0f - z[j]) * h_prev[j] + z[j] * c[j];
  }
}

void GruLayer::backward(const float* x, const float* h_prev, const float* h_out,
                        const float* grad_h, float* grad_p,
                        float* grad_h_prev) const {
  (void)h_out;  // forward pass is recomputed deterministically
  ensureScratch();
  const int in = in_, h = h_;
  // Recompute forward intermediates (deterministic).
  std::vector<float>& z = bz2_;
  std::vector<float>& r = br2_;
  std::vector<float>& c = bc2_;
  std::vector<float>& az = baz_;
  std::vector<float>& ar = bar_;
  std::vector<float>& ac = bac_;
  for (int j = 0; j < h; ++j) {
    az[j] = bz_[j];
    ar[j] = br_[j];
    ac[j] = bh_[j];
    for (int i = 0; i < in; ++i) {
      az[j] += wz_[static_cast<size_t>(i) * h + j] * x[i];
      ar[j] += wr_[static_cast<size_t>(i) * h + j] * x[i];
      ac[j] += wh_[static_cast<size_t>(i) * h + j] * x[i];
    }
    for (int k = 0; k < h; ++k) {
      az[j] += uz_[static_cast<size_t>(k) * h + j] * h_prev[k];
      ar[j] += ur_[static_cast<size_t>(k) * h + j] * h_prev[k];
    }
    z[j] = sigmoid(az[j]);
    r[j] = sigmoid(ar[j]);
  }
  for (int j = 0; j < h; ++j) {
    for (int k = 0; k < h; ++k) {
      ac[j] += uh_[static_cast<size_t>(k) * h + j] * (r[k] * h_prev[k]);
    }
    c[j] = std::tanh(ac[j]);
  }

  // dL/dc, dL/dz
  std::vector<float>& dc = bdc_;
  std::vector<float>& dz = bdz_;
  std::vector<float>& da = bda_;
  std::vector<float>& dr = bdr_;
  std::vector<float>& dlz = bdlz_;
  std::vector<float>& dlr = bdlr_;
  for (int j = 0; j < h; ++j) {
    dc[j] = z[j] * grad_h[j];
    dz[j] = (c[j] - h_prev[j]) * grad_h[j];
    da[j] = dc[j] * (1.0f - c[j] * c[j]);
    // dL/dr = Uh^T da (computed after da known; do via loop below)
  }
  // dL/dr = Uh^T da, elementwise * h_prev
  for (int j = 0; j < h; ++j) {
    float s = 0.0f;
    for (int k = 0; k < h; ++k) {
      s += uh_[static_cast<size_t>(j) * h + k] * da[k];
    }
    dr[j] = s * h_prev[j];
  }
  for (int j = 0; j < h; ++j) {
    dlz[j] = dz[j] * z[j] * (1.0f - z[j]);
    dlr[j] = dr[j] * r[j] * (1.0f - r[j]);
  }

  // Gradient w.r.t. h_prev (full recurrent flow; see gru.h for the formula).
  // Weight layout: u_[k*h+j] connects recurrent unit k to gate j (see the
  // parameter-gradient loops below), so the transpose sums iterate gate j
  // over the per-unit weights.
  if (grad_h_prev) {
    for (int k = 0; k < h; ++k) {
      float s = grad_h[k] * (1.0f - z[k]);
      for (int j = 0; j < h; ++j) {
        s += uz_[static_cast<size_t>(k) * h + j] * dlz[j];
        s += ur_[static_cast<size_t>(k) * h + j] * dlr[j];
        s += uh_[static_cast<size_t>(k) * h + j] * da[j] * r[k];
      }
      grad_h_prev[k] = s;
    }
  }

  // Accumulate gradients into grad_p (layout order: Wz, Uz, bz, Wr, Ur, br,
  // Wh, Uh, bh).
  size_t o = 0;
  auto clearSection = [&](size_t n) {
    std::fill(grad_p + o, grad_p + o + n, 0.0f);
    o += n;
  };
  const size_t n_wi = static_cast<size_t>(in) * h;
  const size_t n_wr = static_cast<size_t>(h) * h;
  clearSection(n_wi);  // Wz
  clearSection(n_wr);  // Uz
  clearSection(h);     // bz
  clearSection(n_wi);  // Wr
  clearSection(n_wr);  // Ur
  clearSection(h);     // br
  clearSection(n_wi);  // Wh
  clearSection(n_wr);  // Uh
  clearSection(h);     // bh
  o = 0;

  // Wz, Uz, bz
  for (int j = 0; j < h; ++j) {
    for (int i = 0; i < in; ++i)
      grad_p[o + static_cast<size_t>(i) * h + j] = dlz[j] * x[i];
  }
  o += n_wi;
  for (int j = 0; j < h; ++j) {
    for (int k = 0; k < h; ++k)
      grad_p[o + static_cast<size_t>(k) * h + j] = dlz[j] * h_prev[k];
  }
  o += n_wr;
  for (int j = 0; j < h; ++j) grad_p[o + j] = dlz[j];
  o += h;

  // Wr, Ur, br
  for (int j = 0; j < h; ++j) {
    for (int i = 0; i < in; ++i)
      grad_p[o + static_cast<size_t>(i) * h + j] = dlr[j] * x[i];
  }
  o += n_wi;
  for (int j = 0; j < h; ++j) {
    for (int k = 0; k < h; ++k)
      grad_p[o + static_cast<size_t>(k) * h + j] = dlr[j] * h_prev[k];
  }
  o += n_wr;
  for (int j = 0; j < h; ++j) grad_p[o + j] = dlr[j];
  o += h;

  // Wh, Uh, bh
  for (int j = 0; j < h; ++j) {
    for (int i = 0; i < in; ++i)
      grad_p[o + static_cast<size_t>(i) * h + j] = da[j] * x[i];
  }
  o += n_wi;
  for (int j = 0; j < h; ++j) {
    for (int k = 0; k < h; ++k)
      grad_p[o + static_cast<size_t>(k) * h + j] = da[j] * (r[k] * h_prev[k]);
  }
  o += n_wr;
  for (int j = 0; j < h; ++j) grad_p[o + j] = da[j];
}

}  // namespace sir
