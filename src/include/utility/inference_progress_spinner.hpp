#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>


extern std::atomic<int> current_layer_idx;
extern std::atomic<bool> running_inference;
extern std::vector<char> spinner_chars;

/**
 * @brief Runs in a separate thread during inference and outputs a nice
 * looking spinner and percentage indicator to standard output
 */
void inference_spinner_function(int total_nodes);