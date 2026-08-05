#include "organism/development.h"

#include <algorithm>

namespace sir {

Development::Development(const DevParams& p) : p_(p) {}

double Development::ageNorm(uint64_t lifetime_steps) const {
  if (p_.maturity_steps == 0) return 1.0;
  return std::min(1.0, static_cast<double>(lifetime_steps) /
                           static_cast<double>(p_.maturity_steps));
}

double Development::learningRateScale(uint64_t lifetime_steps) const {
  double s = 1.0 - p_.lr_age_scale * ageNorm(lifetime_steps);
  return std::max(0.1, s);
}

double Development::epsilonScale(uint64_t lifetime_steps) const {
  double s = 1.0 - p_.epsilon_age_scale * ageNorm(lifetime_steps);
  return std::max(0.05, s);
}

double Development::plasticityScale(uint64_t lifetime_steps) const {
  double s = 1.0 - p_.plasticity_age_scale * ageNorm(lifetime_steps);
  return std::max(0.1, s);
}

}  // namespace sir
