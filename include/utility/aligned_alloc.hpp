#pragma once

#include <cstdlib>
#include <memory>
#include <stdexcept>

/**
 * @brief Function responsible for allocating aligned memory.
 * Instead of allocating an exact amount of memory according to T and data_size
 * memory is padded to be divisable by the provided alignment factorm this
 * ensures that vectorized loads and stores can be safely used without going
 * beyond memory boundraries
 *
 * @return shared_ptr to aligned memory for T[]
 *
 * @author Tim Carlsson (timca@chalmers.se)
 */
template <typename T>
std::shared_ptr<T[]> alloc_aligned_memory(size_t data_size) {
  size_t alignment = MEMORY_ALIGNMENT;

  size_t total_bytes = sizeof(T) * data_size;
  // Rounds up to the nearest value divisiable by the alignment factor
  size_t padded_bytes = ((total_bytes + alignment - 1) / alignment) * alignment;

  void *ptr = nullptr;
  if (posix_memalign(&ptr, alignment, padded_bytes) != 0) {
    throw std::bad_alloc();
  }

  return std::shared_ptr<T[]>(static_cast<T *>(ptr), [](T *ptr) { free(ptr); });
}
