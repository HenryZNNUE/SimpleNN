#include <omp.h>

#include <thread>

#include "nn_adam.h"

int main()
{
    int max_threads = std::thread::hardware_concurrency();
    omp_set_num_threads(max_threads);

    Network nn(INPUT, L1, L2, OUT);
    Trainer trainer;

    trainer.train(nn, "");

    std::cin.get();
    return 0;
}