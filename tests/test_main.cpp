// Test suite entry point. Build and run via ./run.sh --test or
// build/sir_tests [filter]. Never needs SDL.

#include "test_framework.h"

#include <cstdio>

int main(int argc, char** argv) {
  const char* filter = nullptr;
  if (argc > 1) filter = argv[1];
  return sir_test::runAll(filter);
}
