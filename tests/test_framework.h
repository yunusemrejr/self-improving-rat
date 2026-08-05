#pragma once
// Tiny single-header test framework for sir_tests. No external dependencies.
// Each test is a function returning bool (true = pass). A global registry is
// built with static initializers; main() runs every registered test and
// reports failures with a non-zero exit code.

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace sir_test {

// Set to true by a failing CHECK; reset per test by the runner. This makes
// CHECK failures fail the enclosing test even if the return value is ignored.
inline bool& failedFlag() {
  static bool f = false;
  return f;
}

struct Case {
  const char* name;
  std::function<bool()> fn;
};

inline std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}

struct Registrar {
  Registrar(const char* name, std::function<bool()> fn) {
    registry().push_back({name, std::move(fn)});
  }
};

// Checks a condition and prints the failure message with file:line context.
inline bool check(bool ok, const char* expr, const char* file, int line) {
  if (!ok) {
    failedFlag() = true;
    std::fprintf(stderr, "    FAIL %s:%d: %s\n", file, line, expr);
  }
  return ok;
}

inline int runAll() {
  int failed = 0;
  int ran = 0;
  for (const Case& c : registry()) {
    ++ran;
    failedFlag() = false;
    c.fn();
    const bool ok = !failedFlag();
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", c.name);
    if (!ok) ++failed;
  }
  std::printf("\n%d/%d tests passed\n", ran - failed, ran);
  return failed == 0 ? 0 : 1;
}

}  // namespace sir_test

#define CHECK(cond) ::sir_test::check((cond), #cond, __FILE__, __LINE__)
#define CHECK_MSG(cond, msg)                                                  \
  do {                                                                        \
    if (!(cond)) {                                                            \
      ::sir_test::failedFlag() = true;                                        \
      std::fprintf(stderr, "    FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__,   \
                   #cond, (msg));                                             \
    }                                                                         \
  } while (0)
#define TEST(name)                                                            \
  static bool test_##name();                                                  \
  static ::sir_test::Registrar reg_##name(#name, test_##name);                \
  static bool test_##name()
