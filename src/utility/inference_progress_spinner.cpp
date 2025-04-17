#include "utility/inference_progress_spinner.hpp"

std::atomic<int> current_layer_idx{0};
std::atomic<bool> running_inference{true};
std::vector<char> spinner_chars = {'|', '/', '-', '\\'};
std::vector<std::string> dot_animation = {".", "..", "..."};

void inference_spinner_function(int total_nodes) {
    int char_index = 0;
    while (running_inference.load()) {
        char spinner_symbol = spinner_chars[char_index++ % spinner_chars.size()];
        int node_num = current_layer_idx.load();
        
        float progress = 100.0f * node_num / total_nodes;
        std::cout << "\r" << " " << spinner_symbol
                  << " [ " << int(progress) << "% ]"
                  << " Calculating"
                  << std::flush;

        std::this_thread::sleep_for(std::chrono::milliseconds(80));
    }
    std::cout << " \r✓ [ 100% Done ]                      " << std::endl;
}