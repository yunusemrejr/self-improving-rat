#pragma once
// Minimal single-threaded file logger with size-based rotation.
// Keeps at most two files: rat.log and rat.log.1. Bounded by log_max_bytes.

#include <cstdint>
#include <cstdio>
#include <string>

namespace sir {

class Logger {
 public:
  Logger() = default;
  ~Logger();
  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  // Opens (append) the log file; rotates it first if it exceeds max_bytes.
  bool open(const std::string& path, int64_t max_bytes);

  void info(const std::string& msg);
  void warn(const std::string& msg);
  void error(const std::string& msg);
  void log(const std::string& level, const std::string& msg);

 private:
  void rotateIfNeeded();
  std::string timestamp() const;

  FILE* f_ = nullptr;
  std::string path_;
  int64_t max_bytes_ = 0;
  int64_t size_ = 0;
};

}  // namespace sir
