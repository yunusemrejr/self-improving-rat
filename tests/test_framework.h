#pragma once
// Minimal dependency-free test framework for the Self Improving Rat test
// suite. Tests never need SDL; they link against sir_core only.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace sir_test {

struct TestCase {
  const char* name;
  void (*fn)();
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

struct Registrar {
  Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

inline int g_checks = 0;
inline int g_failures = 0;

inline void report_failure(const char* file, int line, const std::string& msg) {
  ++g_failures;
  std::printf("  FAIL %s:%d: %s\n", file, line, msg.c_str());
}

inline int runAll(const char* filter) {
  int ran = 0;
  for (const auto& t : registry()) {
    if (filter && filter[0] && !std::strstr(t.name, filter)) continue;
    const int before = g_failures;
    std::printf("== %s\n", t.name);
    t.fn();
    ++ran;
    if (g_failures != before) {
      std::printf("   -> %d new failure(s)\n", g_failures - before);
    }
  }
  std::printf("\n%d test(s) run, %d check(s), %d failure(s)\n", ran, g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}

}  // namespace sir_test

#define CHECK(cond)                                                       \
  do {                                                                    \
    ++sir_test::g_checks;                                                 \
    if (!(cond)) sir_test::report_failure(__FILE__, __LINE__, #cond);     \
  } while (0)

#define CHECK_MSG(cond, msg)                                              \
  do {                                                                    \
    ++sir_test::g_checks;                                                 \
    if (!(cond)) sir_test::report_failure(__FILE__, __LINE__,             \
                                          std::string(#cond) + " :: " + msg); \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                             \
  do {                                                                    \
    ++sir_test::g_checks;                                                 \
    const double va = (a), vb = (b);                                      \
    if (!(std::fabs(va - vb) <= (tol))) {                                 \
      char buf[160];                                                      \
      std::snprintf(buf, sizeof(buf), "CHECK_NEAR(%s, %s): %.8g vs %.8g", \
                    #a, #b, va, vb);                                      \
      sir_test::report_failure(__FILE__, __LINE__, buf);                  \
    }                                                                     \
  } while (0)

#define TEST(name)                                                        \
  static void test_##name();                                              \
  static ::sir_test::Registrar reg_##name(#name, &test_##name);           \
  static void test_##name()
