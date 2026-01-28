#pragma once

#include <chrono>
#include <iostream>
#include <vector>

#include "mnist_reader.h"
#include "terminal.h"

constexpr int INPUT = 784;
constexpr int L1 = 384;
constexpr int L2 = 64;
constexpr int OUT = 10;

constexpr float learning_rate = 0.001;

constexpr int batch_size = 32;
constexpr int epoch_size = 100;

constexpr float beta1 = 0.9f;
constexpr float beta2 = 0.999f;
constexpr float eps = 1e-8f;

enum InitType { XAVIER, HE };

struct Layer
{
    std::vector<float> weights;
    std::vector<float> biases;
    std::vector<float> values;
    std::vector<float> grads;
    std::vector<float> gradsW;
    std::vector<float> mW, vW;
    std::vector<float> mB, vB;
    int n_in = 0, n_out = 0;

    Layer(int dim1, int dim2, InitType type = HE);
};

struct Progress
{
    std::chrono::steady_clock::time_point last_update = std::chrono::steady_clock::now();
    int bar_width = 50;
    bool unix = get_unix();

    void update(int epoch, int total_epochs, int batch_idx, int batches_in_epoch, float loss, float acc)
    {
        auto now = std::chrono::steady_clock::now();

        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_update).count() < 1)
        {
            return;
        }

        last_update = now;

        float progress = static_cast<float>(batch_idx) / batches_in_epoch;
        if (progress > 1.0f)
            progress = 1.0f;
        int pos = static_cast<int>(progress * bar_width);

        std::ostringstream line;
        line << Blue(unix) << "Epoch " << epoch << "/" << total_epochs << Vanilla(unix) << " [";

        for (int i = 0; i < bar_width; ++i)
        {
            if (i <= pos)
            {
                line << WhiteBG(unix) << " ";
            } else
            {
                line << Vanilla(unix) << " ";
            }
        }

        line << "] " << Blue(unix) << std::setw(3) << static_cast<int>(progress * 100.0f) << "% | Loss: " << std::fixed
             << std::setprecision(4) << loss << " | Acc: " << std::fixed << std::setprecision(2) << acc * 100.0f << "%"
             << Vanilla(unix);

        std::cout << "\r" << line.str() << std::string(20, ' ') << std::flush;
    }

    void end_epoch(int epoch, int total_epochs, float loss, float acc)
    {
        std::ostringstream line;
        line << Blue(unix) << "Epoch " << epoch << "/" << total_epochs << Vanilla(unix) << " [" << WhiteBG(unix)
             << std::string(bar_width, ' ') << Vanilla(unix) << "]" << Blue(unix) << " 100% "
             << "| Loss: " << std::fixed << std::setprecision(4) << loss << " | Acc: " << std::fixed
             << std::setprecision(2) << acc * 100.0f << "%";

        std::cout << "\r" << line.str() << std::string(20, ' ') << std::endl;
    }
};

class Network
{
    std::vector<Layer> layers;
    InitType type = HE;

public:
    int forward(std::vector<float>& input);
    void backward(std::vector<float>& input, uint8_t& label, float& loss, float& lr);
    void apply_grads(int& batch, float& lr, int& t);

    void reset_values();
    void zero_grads();

    void load(std::string path);
    void save(std::string path = "mnist.nn");

    template <typename... Args>
    Network(Args... sizes) : Network({static_cast<int>(sizes)...})
    {
    }

    Network(std::initializer_list<int> sizes);
};

class Trainer
{
    mnist::MNIST_dataset<uint8_t, uint8_t> dataset;

public:
    void train(Network& network, std::string path = "mnist.nn", int epochs = epoch_size, int batch = batch_size,
               float lr = learning_rate);

    Trainer() : dataset(mnist::read_dataset<>()) {}
};