#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>

#include <modularml>

std::vector<std::vector<uint8_t>> load_mnist_images(std::string path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file" + path);
    }

    uint32_t magic_number = 0;
    uint32_t num_images = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;

    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&num_rows), 4);
    file.read(reinterpret_cast<char*>(&num_cols), 4);

    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);

    if (magic_number != 2051) {
        throw std::runtime_error("Invalid MNIST image file!");
    }

    std::vector<std::vector<uint8_t>> images(num_images, std::vector<uint8_t>(num_rows * num_cols));

    for (auto& image : images) {
        file.read(reinterpret_cast<char*>(image.data()), num_rows * num_cols);
    }

    return images;
}

std::vector<uint8_t> load_mnist_labels(std::string path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file" + path);
    }

    uint32_t magic_number = 0;
    uint32_t num_labels = 0;

    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);

    magic_number = __builtin_bswap32(magic_number);
    num_labels = __builtin_bswap32(num_labels);

    if (magic_number != 2049) {
        throw std::runtime_error("Invalid MNIST labels file!");
    }
    
    std::vector<uint8_t> labels(num_labels);

    file.read(reinterpret_cast<char*>(labels.data()), num_labels);
    
    return labels;
}

std::shared_ptr<Tensor<float>> image_to_tensor(std::vector<uint8_t> image) {
    array_mml<unsigned long int> image_tensor_shape({1, 1, 28, 28 });
    array_mml<float> output_data(28 * 28);

    for (int i=0; i < image.size(); i++) {
        output_data[i] = static_cast<float>(image[i]) / 255.0;
    }

    std::shared_ptr<Tensor<float>> output = TensorFactory::create_tensor<float>(image_tensor_shape, output_data);

    return output;
}

/**
 * @brief Runs an input image through the provided model
 * 
 * @return Returns a vector containing indexes to the top 5 predictions
 */
std::vector<int> run_lenet_inference(const std::unique_ptr<Model>& model, std::vector<uint8_t> image) {
    
    std::unordered_map<std::string, GeneralDataTypes> inputs;
    std::unordered_map<std::string, GeneralDataTypes> outputs;
    
    try {   
        auto image_tensor = image_to_tensor(image);
        inputs["input"] = image_tensor;
        outputs = model->infer(inputs);

    } catch(const std::exception& e){
        std::cerr << "Inference failed: " << e.what() << '\n';
    }
    if (outputs.empty()) {
        std::cerr << "Error: output empty" << std::endl;
    }
    // Sequential is something i found during debug seems to be different key with different frameworks that export onnx
    auto prediction = outputs.find("sequential");
    
    if (prediction == outputs.end()) {
        throw std::runtime_error("Demo: Output tensor not found in iomap");
    }
    auto output_tensor = std::get<std::shared_ptr<Tensor<float>>>(prediction->second);
    int max_index = TensorOperations::arg_max<float>(output_tensor);
    std::cout << "Prediction: " << max_index << std::endl;
    
    return std::vector<int>();
}

/**
 * @brief Load a test test of mnist images and labels.
 * We use this to validate the Lenet model ModularML constructs internally
 */
int main(int argc, char* argv[]) {
    std::string model_path = argv[1];
    
    auto images = load_mnist_images("demo/lenet_demo/t10k-images.idx3-ubyte");
    auto labels = load_mnist_labels("demo/lenet_demo/t10k-labels.idx1-ubyte");

    std::cout << "Loaded images: " << images.size() << std::endl;
    std::cout << "Loaded labels: " << images.size() << std::endl;
    
    nlohmann::json json_model;
    std::ifstream file(model_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open json file");
        return 1;
    }

    file >> json_model;
    file.close();

    Parser_mml parser;
    std::unique_ptr<Model> model;


    Profiler::begin_timing("Parsing AlexNet");
    // Try to parse the model provided
    try {
        model = parser.parse(json_model);
        std::cout << "Successfully loaded model" << std::endl;
    }
    catch(const std::exception& e) {
        std::cerr << "Error occurred when trying to parse json: " << e.what() << '\n';
        return 1;
    }

    auto prediction = run_lenet_inference(model, images[0]);
    return 0;
}