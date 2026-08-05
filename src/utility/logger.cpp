#include "utility/logger.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <sys/stat.h>
#include <unistd.h>

namespace sir {

Logger::~Logger() {
  if (f_) fclose(f_);
}

bool Logger::open(const std::string& path, int64_t max_bytes) {
  path_ = path;
  max_bytes_ = max_bytes;

  // Rotate an oversized pre-existing log once at open time.
  struct stat st;
  if (stat(path_.c_str(), &st) == 0 && st.st_size > max_bytes_) {
    std::string old = path_ + ".1";
    ::remove(old.c_str());
    ::rename(path_.c_str(), old.c_str());
  }

  f_ = fopen(path_.c_str(), "a");
  if (!f_) return false;
  if (fseek(f_, 0, SEEK_END) == 0) size_ = static_cast<int64_t>(ftell(f_));
  return true;
}

std::string Logger::timestamp() const {
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&t, &tm);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d",
                tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour,
                tm.tm_min, tm.tm_sec, static_cast<int>(ms.count()));
  return buf;
}

void Logger::rotateIfNeeded() {
  if (!f_ || size_ < max_bytes_) return;
  fclose(f_);
  f_ = nullptr;
  std::string old = path_ + ".1";
  ::remove(old.c_str());
  ::rename(path_.c_str(), old.c_str());
  f_ = fopen(path_.c_str(), "a");
  size_ = 0;
  if (!f_) return;
  std::string head = "[" + timestamp() + "] [INFO] log rotated (was " + old + ")\n";
  fputs(head.c_str(), f_);
  size_ += static_cast<int64_t>(head.size());
}

void Logger::log(const std::string& level, const std::string& msg) {
  if (!f_) return;
  rotateIfNeeded();
  std::string line = "[" + timestamp() + "] [" + level + "] " + msg + "\n";
  fputs(line.c_str(), f_);
  fflush(f_);
  size_ += static_cast<int64_t>(line.size());
}

void Logger::info(const std::string& msg) { log("INFO", msg); }
void Logger::warn(const std::string& msg) { log("WARN", msg); }
void Logger::error(const std::string& msg) { log("ERROR", msg); }

}  // namespace sir
