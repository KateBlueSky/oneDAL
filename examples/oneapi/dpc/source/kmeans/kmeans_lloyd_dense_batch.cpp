#include <iostream>
#include <oneapi/dal/algo/kmeans.hpp>
#include <oneapi/dal/table/homogen.hpp>
#include <random>
#include <chrono>

namespace dal = oneapi::dal;

int main(int argc, char* argv[]) {
    // Parameters for data generation
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <row_count>" << std::endl;
        return 1;
    }


    const std::int64_t row_count = std::atoi(argv[1]); // Number of samples
    const std::int64_t column_count = 100; // Number of features (similar to Higgs dataset)
    const std::int64_t cluster_count = 20; // Number of clusters

    // Random data generation
    std::mt19937 rng(42); // Random number generator with a fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    // Generate training data
    std::vector<float> train_data(row_count * column_count);
    for (auto& value : train_data) {
        value = dist(rng);
    }
     const auto x_train = dal::homogen_table::wrap(train_data.data(), row_count, column_count);

    // Generate initial centroids (randomly select some points from the training data)
    std::vector<float> initial_centroids(cluster_count * column_count);
    for (std::int64_t i = 0; i < cluster_count; ++i) {
        std::copy_n(train_data.begin() + i * column_count, column_count, initial_centroids.begin() + i * column_count);
    }
    const auto initial_centroids_table = dal::homogen_table::wrap(initial_centroids.data(),cluster_count, column_count);

    // Generate test data
    std::vector<float> test_data(row_count * column_count);
    for (auto& value : test_data) {
        value = dist(rng);
    }
    const auto x_test = dal::homogen_table::wrap(test_data.data(), row_count, column_count);

    // Generate test labels (randomly for demonstration purposes)
    std::vector<float> test_labels(row_count);
    for (auto& label : test_labels) {
        label = static_cast<float>(rng() % cluster_count);
    }
    const auto y_test = dal::homogen_table::wrap(test_labels.data(),row_count, 1);

    // K-means algorithm setup
    const auto kmeans_desc = dal::kmeans::descriptor<>()
                                 .set_cluster_count(cluster_count)
                                 .set_max_iteration_count(5)
                                 .set_accuracy_threshold(0.001);

    // Train the model
    auto start = std::chrono::high_resolution_clock::now();
    const auto result_train = dal::train(kmeans_desc, x_train, initial_centroids_table);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << row_count << "," << elapsed.count() << "\n";


    //std::cout << "Iteration count: " << result_train.get_iteration_count() << std::endl;
    //std::cout << "Objective function value: " << result_train.get_objective_function_value() << std::endl;
    //std::cout << "Responses:\n" << result_train.get_responses() << std::endl;
    //std::cout << "Centroids:\n" << result_train.get_model().get_centroids() << std::endl;

    // Infer using the trained model
    const auto result_test = dal::infer(kmeans_desc, result_train.get_model(), x_test);

    //std::cout << "Infer result:\n" << result_test.get_responses() << std::endl;
    //std::cout << "Ground truth:\n" << y_test << std::endl;

    return 0;
}

