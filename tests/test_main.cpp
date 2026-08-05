// Test suite entry point. Build and run via ./run.sh --test or
// build/sir_tests [filter]. Never needs SDL.

#include "test_framework.h"

#include <cstdio>

int main(int argc, char** argv) {
  // Line-buffer stdout so test progress is visible even when piped (e.g.
  // via ./run.sh --test); the default block buffer swallows output until
  // the suite finishes, which makes a slow suite look like a hang.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const char* filter = nullptr;
  if (argc > 1) filter = argv[1];
  return sir_test::runAll(filter);
}
