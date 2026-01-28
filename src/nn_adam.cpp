#include "nn_adam.h"

#include <omp.h>

#include <algorithm>
#include <fstream>
#include <random>
#include <thread>

Layer::Layer(int dim1, int dim2, InitType type)
{
    n_in = dim1;
    n_out = dim2;

    weights.resize(n_in * n_out);
    biases.assign(n_out, 0.01f);
    values.assign(biases.begin(), biases.end());
    grads.assign(n_out, 0.0f);
    gradsW.assign(n_in * n_out, 0.0f);
    mW.resize(n_in * n_out, 0.0f);
    vW.resize(n_in * n_out, 0.0f);
    mB.resize(n_out, 0.0f);
    vB.resize(n_out, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());

    if (type == XAVIER)
    {
        float limit = std::sqrt(6.0 / (n_in + n_out));
        std::uniform_real_distribution<> dist(-limit, limit);

        for (int i = 0; i < dim1 * dim2; ++i)
        {
            weights[i] = dist(gen) * 0.1;
        }
    } else if (type == HE)
    {
        float stddev = std::sqrt(2.0 / n_in);
        std::normal_distribution<> dist(0.0, stddev);

        for (int i = 0; i < dim1 * dim2; ++i)
        {
            weights[i] = dist(gen);
        }
    }
}

int Network::forward(std::vector<float>& input)
{
    reset_values();

    /*
    std::vector<float> in(layers[0].n_in);

#pragma omp parallel for
    for (int j = 0; j < layers[0].n_in; ++j)
    {
            in[j] = static_cast<float>(input[j]) / 255.0f;
    }
    */

#pragma omp parallel for
    for (int i = 0; i < layers[0].n_out; ++i)
    {
        for (int j = 0; j < layers[0].n_in; ++j)
        {
            layers[0].values[i] += layers[0].weights[i * layers[0].n_in + j] * input[j];
        }

        layers[0].values[i] = layers[0].values[i] > 0 ? layers[0].values[i] : 0;
    }

    for (int i = 1; i < layers.size() - 1; ++i)
    {
#pragma omp parallel for
        for (int j = 0; j < layers[i].n_out; ++j)
        {
#pragma omp simd
            for (int k = 0; k < layers[i].n_in; ++k)
            {
                layers[i].values[j] += layers[i].weights[j * layers[i].n_in + k] * layers[i - 1].values[k];
            }

            layers[i].values[j] = layers[i].values[j] > 0 ? layers[i].values[j] : 0;
        }
    }

    for (int i = 0; i < layers.back().n_out; ++i)
    {
        for (int j = 0; j < layers.back().n_in; ++j)
        {
            layers.back().values[i] +=
                layers.back().weights[i * layers.back().n_in + j] * layers[layers.size() - 2].values[j];
        }
    }

    float max_val = *std::max_element(layers.back().values.begin(), layers.back().values.end());
    float sum = 0.0f;

#pragma omp simd reduction(+ : sum)
    for (int i = 0; i < layers.back().values.size(); ++i)
    {
        layers.back().values[i] = std::exp(layers.back().values[i] - max_val);
        sum += layers.back().values[i];
    }

#pragma omp simd
    for (float& v : layers.back().values)
    {
        v /= sum;
    }

    return std::distance(layers.back().values.begin(),
                         std::max_element(layers.back().values.begin(), layers.back().values.end()));
}

void Network::backward(std::vector<float>& input, uint8_t& label, float& loss, float& lr)
{
    std::vector<float> target(layers.back().n_out, 0.0f);
    target[label] = 1.0f;
    float epsilon = 1e-9f;
    loss = 0.0f;

#pragma omp simd reduction(+ : loss)
    for (int i = 0; i < layers.back().n_out; ++i)
    {
        loss += -target[i] * std::log(std::clamp(layers.back().values[i], epsilon, 1.0f - epsilon));
    }

    /*
    for (int i = 0; i < OUT; ++i)
    {
            loss += std::pow(layers.back().values[i] - target[i], 2);
    }

    loss /= OUT;
    */

#pragma omp parallel for
    for (int i = 0; i < layers.back().n_out; ++i)
    {
        float grad = layers.back().values[i] - target[i];
        layers.back().grads[i] += grad;

#pragma omp simd
        for (int j = 0; j < layers.back().n_in; ++j)
        {
            layers.back().gradsW[i * layers.back().n_in + j] += grad * layers[layers.size() - 2].values[j];
        }
    }

    for (int i = layers.size() - 2; i >= 1; --i)
    {
#pragma omp parallel for
        for (int j = 0; j < layers[i].n_out; ++j)
        {
            float grad_sum = 0.0f;

#pragma omp simd reduction(+ : grad_sum)
            for (int k = 0; k < layers[i + 1].n_out; ++k)
            {
                grad_sum += layers[i + 1].weights[k * layers[i + 1].n_in + j] * layers[i + 1].grads[k];
            }

            float delta = (layers[i].values[j] > 0 ? 1.0f : 0.0f) * grad_sum;
            layers[i].grads[j] += delta;

#pragma omp simd
            for (int k = 0; k < layers[i].n_in; ++k)
            {
                layers[i].gradsW[j * layers[i].n_in + k] += delta * layers[i - 1].values[k];
            }
        }
    }

    /*
    std::vector<float> in(layers[0].n_in);

#pragma omp parallel for
    for (int j = 0; j < layers[0].n_in; ++j)
    {
            in[j] = static_cast<float>(input[j]) / 255.0f;
    }
    */

#pragma omp parallel for
    for (int i = 0; i < layers[0].n_out; ++i)
    {
        float delta = layers[0].grads[i];

#pragma omp simd
        for (int j = 0; j < layers[0].n_in; ++j)
        {
            layers[0].gradsW[i * layers[0].n_in + j] += delta * input[j];
        }
    }
}

void Network::apply_grads(int& batch, float& lr, int& t)
{
    float bc1 = 1.0f - std::pow(beta1, t);
    float bc2 = 1.0f - std::pow(beta2, t);

#pragma omp parallel for
    for (int i = 0; i < layers.back().n_out; ++i)
    {
#pragma omp simd
        for (int j = 0; j < layers.back().n_in; ++j)
        {
            float g = layers.back().gradsW[i * layers.back().n_in + j] / batch;

            layers.back().mW[i * layers.back().n_in + j] =
                beta1 * layers.back().mW[i * layers.back().n_in + j] + (1 - beta1) * g;
            layers.back().vW[i * layers.back().n_in + j] =
                beta2 * layers.back().vW[i * layers.back().n_in + j] + (1 - beta2) * g * g;

            float m_hat = layers.back().mW[i * layers.back().n_in + j] / bc1;
            float v_hat = layers.back().vW[i * layers.back().n_in + j] / bc2;
            layers.back().weights[i * layers.back().n_in + j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }

#pragma omp simd
    for (int i = 0; i < layers.back().n_out; ++i)
    {
        float g = layers.back().grads[i] / batch;

        layers.back().mB[i] = beta1 * layers.back().mB[i] + (1 - beta1) * g;
        layers.back().vB[i] = beta2 * layers.back().vB[i] + (1 - beta2) * g * g;

        float m_hat = layers.back().mB[i] / bc1;
        float v_hat = layers.back().vB[i] / bc2;

        layers.back().biases[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }

    for (int i = layers.size() - 2; i >= 0; --i)
    {
#pragma omp parallel for
        for (int j = 0; j < layers[i].n_out; ++j)
        {
#pragma omp simd
            for (int k = 0; k < layers[i].n_in; ++k)
            {
                float g = layers[i].gradsW[j * layers[i].n_in + k] / batch;

                layers[i].mW[j * layers[i].n_in + k] = beta1 * layers[i].mW[j * layers[i].n_in + k] + (1 - beta1) * g;
                layers[i].vW[j * layers[i].n_in + k] =
                    beta2 * layers[i].vW[j * layers[i].n_in + k] + (1 - beta2) * g * g;

                float m_hat = layers[i].mW[j * layers[i].n_in + k] / bc1;
                float v_hat = layers[i].vW[j * layers[i].n_in + k] / bc2;

                layers[i].weights[j * layers[i].n_in + k] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        }

#pragma omp simd
        for (int k = 0; k < layers[i].n_out; ++k)
        {
            float g = layers[i].grads[k] / batch;

            layers[i].mB[k] = beta1 * layers[i].mB[k] + (1 - beta1) * g;
            layers[i].vB[k] = beta2 * layers[i].vB[k] + (1 - beta2) * g * g;

            float m_hat = layers[i].mB[k] / bc1;
            float v_hat = layers[i].vB[k] / bc2;

            layers[i].biases[k] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
}

void Network::reset_values()
{
    for (Layer& layer : layers)
    {
        layer.values = layer.biases;
    }
}

void Network::zero_grads()
{
    for (Layer& layer : layers)
    {
        std::fill(layer.grads.begin(), layer.grads.end(), 0.0f);
        std::fill(layer.gradsW.begin(), layer.gradsW.end(), 0.0f);
    }
}

void Network::load(std::string path) {}

void Network::save(std::string path)
{
    std::ofstream file(path, std::ios::out | std::ios::trunc | std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Failed to open " << path << " for writing\n";
        return;
    }

    for (Layer& layer : layers)
    {
        file.write(reinterpret_cast<char*>(layer.weights.data()), sizeof(float) * layer.weights.size());
        file.write(reinterpret_cast<char*>(layer.biases.data()), sizeof(float) * layer.biases.size());
    }

    file.close();
}

Network::Network(std::initializer_list<int> sizes)
{
    int buf = 0;
    int index = 0;

    for (auto it = sizes.begin(); it != sizes.end(); ++it, ++index)
    {
        int dim = *it;

        if (index > 0)
        {
            InitType type;

            if (std::next(it) == sizes.end())
            {
                type = XAVIER;
            } else
            {
                type = HE;
            }

            Layer layer(buf, dim, type);
            layers.emplace_back(layer);
        }

        buf = dim;
    }
}

void Trainer::train(Network& network, std::string path, int epochs, int batch, float lr)
{
    const int N = static_cast<int>(dataset.training_images.size());
    Progress progress;
    int t = 1;

    for (int i = 0; i < epochs; ++i)
    {
        int batches_in_epoch = (N + batch - 1) / batch;
        int batch_idx = 0;
        float batch_loss = 0.0f;
        float acc = 0.0f;

        for (int b = 0; b < N; b += batch)
        {
            int b_end = std::min(b + batch, N);
            network.zero_grads();

            int correct = 0;

            for (int idx = b; idx < b_end; ++idx)
            {
                std::vector<uint8_t>& input = dataset.training_images[idx];
                uint8_t& label = dataset.training_labels[idx];
                float sample_loss = 0.0f;

                std::vector<float> in(INPUT);

#pragma omp simd
                for (int j = 0; j < INPUT; ++j)
                {
                    in[j] = static_cast<float>(input[j]) / 255.0f;
                }

                int eval = network.forward(in);
                if (eval == label)
                    ++correct;

                network.backward(in, label, sample_loss, lr);
                batch_loss += sample_loss;
            }

            batch_loss /= (b_end - b);
            int batch_count = b_end - b;

            network.apply_grads(batch_count, lr, t);

            ++t;
            ++batch_idx;
            acc = static_cast<float>(correct) / batch_count;
            progress.update(i + 1, epochs, batch_idx, batches_in_epoch, batch_loss, acc);
        }

        progress.end_epoch(i + 1, epochs, batch_loss, acc);

        if (i > 0 && i % 20 == 0)
        {
            lr *= 0.7f;
        }

        if ((i + 1) % 10 == 0)
        {
            std::string temp_path = "mnist-epoch" + std::to_string(i + 1) + ".nn";
            network.save(temp_path);
        }
    }

    network.save("mnist.nn");
}