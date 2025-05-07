#pragma once

#include <cstdlib>
#include <memory>
#include <stdexcept>

/**
 * @brief Function responsible for allocating aligned memory
 * Instead of allocating an exact amount of memory according to T and data_size
 * memory is padded to be divisible by the provided alignment factor, this ensures
 * that vector loads and stores can be safely used without going beyond the memory boundrary.
 * 
 * @return shared_ptr to aligned memory for T[]
 * 
 * @author Tim Carlsson (timca@chalmers.se)
 * 
 */
template <typename T>
std::shared_ptr<T[]> alloc_aligned_memory(size_t data_size) {
    size_t alignment = MEMORY_ALIGNMENT; // Gets set during compilation

    size_t total_bytes = sizeof(T) * data_size;
    // Rounds up to the nearest value divisable by alignment factor
    size_t padded_bytes = ((total_bytes + alignment - 1) / alignment) * alignment;

    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, padded_bytes) != 0) {
        throw std::bad_alloc();
    }

    // Return the shared_ptr and provide a custom destructor.
    return std::shared_ptr<T[]>(static_cast<T*>(ptr), [](T *ptr) {
        free(ptr);
    });
}