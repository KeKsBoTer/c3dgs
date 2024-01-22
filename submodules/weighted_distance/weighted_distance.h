#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> weightedDistanceCUDA(
    const torch::Tensor& coefs,
    const torch::Tensor& codebook);