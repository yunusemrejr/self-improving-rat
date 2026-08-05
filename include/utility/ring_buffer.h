#pragma once
// Fixed-capacity ring buffer. Preallocated at construction; never grows.
// Used for rolling statistics and any bounded history.

#include <cstddef>
#include <type_traits>
#include <vector>

namespace sir {

template <typename T>
class RingBuffer {
 public:
  explicit RingBuffer(size_t capacity) : data_(capacity), capacity_(capacity) {}

  void push(const T& v) {
    // Ring convention: head_ always points at the OLDEST element; the new
    // element is written at head_ + count_ (mod capacity). When full, the
    // write position is the oldest slot, which then becomes the new head_.
    data_[(head_ + count_) % capacity_] = v;
    if (count_ < capacity_) {
      ++count_;
    } else {
      head_ = (head_ + 1) % capacity_;
    }
  }

  void clear() {
    head_ = 0;
    count_ = 0;
  }

  size_t size() const { return count_; }
  size_t capacity() const { return capacity_; }
  bool empty() const { return count_ == 0; }
  bool full() const { return count_ == capacity_; }

  // Index 0 = oldest element, index size()-1 = newest.
  const T& operator[](size_t i) const { return data_[(head_ + i) % capacity_]; }
  T& operator[](size_t i) { return data_[(head_ + i) % capacity_]; }

  // Sum of the contents (arithmetic types only).
  template <typename U = T,
            typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
  double sum() const {
    double s = 0;
    for (size_t i = 0; i < count_; ++i) s += data_[(head_ + i) % capacity_];
    return s;
  }

 private:
  std::vector<T> data_;
  size_t head_ = 0;
  size_t count_ = 0;
  size_t capacity_;
};

}  // namespace sir
