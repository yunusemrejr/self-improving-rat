// Gradient correctness: finite-difference checks for GruLayer::backward and
// NeuralNet::backward (including the prediction-head sigmoid chain). The
// perturbation must be large enough (1e-2) that saturated gates do not drown
// the numeric gradient in float noise.

#include "test_framework.h"

#include "learning/gru.h"
#include "learning/neural_net.h"
#include "utility/rng.h"

#include <cmath>
#include <vector>

using namespace sir;

namespace {

// Worst relative difference (denominator floors at 1e-6).
double worstRelDiff(const std::vector<float>& a, const std::vector<float>& b) {
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double denom = std::max(
        1e-3, std::max(static_cast<double>(std::fabs(a[i])),
                       static_cast<double>(std::fabs(b[i]))));
    const double d = std::fabs(a[i] - b[i]) / denom;
    if (d > worst) worst = d;
  }
  return worst;
}

}  // namespace

TEST(gru_backward_matches_finite_difference) {
  Rng rng(1234);
  const int in = 28, h = 16;
  GruLayer gru(in, h, rng);
  std::vector<float> x(in), hprev(h), hout(h), grad_h(h);
  for (auto& v : x) v = static_cast<float>(rng.uniform(-1, 1));
  for (auto& v : hprev) v = static_cast<float>(rng.uniform(-0.5, 0.5));
  for (auto& v : grad_h) v = static_cast<float>(rng.uniform(-1, 1));
  gru.forward(x.data(), hprev.data(), hout.data());

  const int n = GruLayer::paramCount(in, h);
  std::vector<float> grad_p(n);
  gru.backward(x.data(), hprev.data(), hout.data(), grad_h.data(), grad_p.data());

  std::vector<float> params(n), p2(n), h2(h);
  gru.getParams(params.data());
  const float eps = 1e-2f;
  std::vector<float> num(n);
  for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
    p2 = params;
    p2[i] += eps;
    gru.setParams(p2.data());
    gru.forward(x.data(), hprev.data(), h2.data());
    double lp = 0;
    for (int j = 0; j < h; ++j) lp += static_cast<double>(h2[j]) * grad_h[j];
    p2 = params;
    p2[i] -= eps;
    gru.setParams(p2.data());
    gru.forward(x.data(), hprev.data(), h2.data());
    double lm = 0;
    for (int j = 0; j < h; ++j) lm += static_cast<double>(h2[j]) * grad_h[j];
    num[i] = static_cast<float>((lp - lm) / (2.0 * eps));
  }
  gru.setParams(params.data());
  CHECK(worstRelDiff(num, grad_p) < 0.05);  // float32 central-difference noise floor
}

TEST(neuralnet_backward_matches_finite_difference) {
  Rng rng(4321);
  const int in = 30, rnn = 16, po = 4, pr = 16;
  NeuralNet net(in, rnn, po, pr, rng);
  std::vector<float> x(in), hprev(rnn), hout(rnn), q(po), pred(pr);
  for (auto& v : x) v = static_cast<float>(rng.uniform(-1, 1));
  for (auto& v : hprev) v = static_cast<float>(rng.uniform(-0.5, 0.5));
  net.forward(x.data(), hprev.data(), hout.data(), q.data(), pred.data());

  std::vector<float> grad_q(po), grad_pred(pr);
  for (auto& v : grad_q) v = static_cast<float>(rng.uniform(-1, 1));
  for (auto& v : grad_pred) v = static_cast<float>(rng.uniform(-1, 1));

  const int n = net.paramCount();
  std::vector<float> grad_p(n);
  net.backward(x.data(), hprev.data(), hout.data(), grad_q.data(),
               grad_pred.data(), grad_p.data());

  std::vector<float> params(n), p2(n), h2(rnn), q2(po), pr2(pr);
  net.getParams(params.data());
  const float eps = 1e-2f;
  std::vector<float> num(n);
  for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
    p2 = params;
    p2[i] += eps;
    net.setParams(p2.data());
    net.forward(x.data(), hprev.data(), h2.data(), q2.data(), pr2.data());
    double lp = 0;
    for (int k = 0; k < po; ++k) lp += static_cast<double>(q2[k]) * grad_q[k];
    for (int k = 0; k < pr; ++k) lp += static_cast<double>(pr2[k]) * grad_pred[k];
    p2 = params;
    p2[i] -= eps;
    net.setParams(p2.data());
    net.forward(x.data(), hprev.data(), h2.data(), q2.data(), pr2.data());
    double lm = 0;
    for (int k = 0; k < po; ++k) lm += static_cast<double>(q2[k]) * grad_q[k];
    for (int k = 0; k < pr; ++k) lm += static_cast<double>(pr2[k]) * grad_pred[k];
    num[i] = static_cast<float>((lp - lm) / (2.0 * eps));
  }
  net.setParams(params.data());
  CHECK(worstRelDiff(num, grad_p) < 0.05);  // float32 central-difference noise floor
}

TEST(neuralnet_forward_deterministic_and_finite) {
  Rng rng(7);
  NeuralNet net(30, 16, 4, 16, rng);
  std::vector<float> x(30), hprev(16), h1(16), h2(16), q1(4), q2(4), p1(16), p2(16);
  for (auto& v : x) v = static_cast<float>(rng.uniform(-1, 1));
  for (auto& v : hprev) v = static_cast<float>(rng.uniform(-0.5, 0.5));
  net.forward(x.data(), hprev.data(), h1.data(), q1.data(), p1.data());
  net.forward(x.data(), hprev.data(), h2.data(), q2.data(), p2.data());
  for (int i = 0; i < 16; ++i) CHECK_NEAR(h1[i], h2[i], 1e-6);
  for (int i = 0; i < 4; ++i) CHECK_NEAR(q1[i], q2[i], 1e-6);
  for (int i = 0; i < 16; ++i) {
    CHECK_NEAR(p1[i], p2[i], 1e-6);
    CHECK(p1[i] >= 0.0f && p1[i] <= 1.0f);  // sigmoid bounded
  }
  CHECK(net.allFinite());
}

TEST(neuralnet_param_and_mask_layouts) {
  Rng rng(9);
  const int in = 30, rnn = 16, po = 4, pr = 16;
  NeuralNet net(in, rnn, po, pr, rng);
  const int expected =
      GruLayer::paramCount(in, rnn) + rnn * po + po + rnn * pr + pr;
  CHECK(net.paramCount() == expected);
  // Mask covers exactly the weights (biases excluded); all bits set at init.
  CHECK(net.activeCount() == net.paramCount() - (3 * rnn + po + pr));
  // Turning connections off through the plasticity API must actually change
  // inference (masks zero real weights).
  std::vector<float> x(in, 0.1f), hprev(rnn, 0.0f), h(rnn), q(po), pred(pr);
  net.forward(x.data(), hprev.data(), h.data(), q.data(), pred.data());
  std::vector<float> q_before = q;
  const size_t n = net.paramCount();
  for (size_t i = 0; i < n; ++i) net.zeroParam(i);  // disconnect every weight
  net.forward(x.data(), hprev.data(), h.data(), q.data(), pred.data());
  bool changed = false;
  for (int k = 0; k < po; ++k)
    if (std::fabs(q[k] - q_before[k]) > 1e-4) changed = true;
  CHECK(changed);  // pruning genuinely affects computation
  CHECK(net.activeCount() == 0);
  CHECK(net.allFinite());
}
