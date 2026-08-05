// Application entry point: argument parsing, environment hooks, signal
// handlers, and the exception boundary around the main loop.

#include "app/application.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

void handleSignal(int) {
  // First signal: request a clean shutdown (the main loop polls the counter,
  // saves a checkpoint and exits). Restore the default handler so a second
  // signal terminates immediately.
  ++sir::g_signal_count;
  std::signal(SIGINT, SIG_DFL);
  std::signal(SIGTERM, SIG_DFL);
}

}  // namespace

int main(int argc, char** argv) {
  sir::Application::Options opts;

  // CLI arguments.
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      opts.config_path = argv[++i];
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      std::printf("usage: %s [--config PATH]\n"
                  "environment: SIR_CONFIG SIR_SEED SIR_HEADLESS=1 "
                  "SIR_MAX_STEPS=N SIR_SCREENSHOT=path.bmp\n",
                  argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s (try --help)\n", argv[i]);
      return 2;
    }
  }

  // Environment hooks (documented in README).
  if (const char* v = std::getenv("SIR_CONFIG")) opts.config_path = v;
  if (const char* v = std::getenv("SIR_HEADLESS")) {
    opts.headless = std::atoi(v) != 0;
  }
  if (const char* v = std::getenv("SIR_SCREENSHOT")) {
    opts.screenshot_path = v;
  }
  if (const char* v = std::getenv("SIR_MAX_STEPS")) {
    opts.max_steps = std::strtoull(v, nullptr, 10);
  }
  if (const char* v = std::getenv("SIR_SEED")) {
    opts.seed_override = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
  }

  // Graceful shutdown on SIGINT/SIGTERM (save checkpoint, then exit).
  std::signal(SIGINT, handleSignal);
  std::signal(SIGTERM, handleSignal);

  try {
    return sir::Application().run(opts);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fatal error: %s\n", e.what());
    return 1;
  } catch (...) {
    std::fprintf(stderr, "fatal error: unknown exception\n");
    return 1;
  }
}
