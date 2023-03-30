#pragma once
#include <torch/torch.h>
//SOURCE: https://gitlab.engr.illinois.edu/jundayu2/nsa-nfm/-/blob/master/src/rllab/rllab/misc/tensor_utils.py
torch::Tensor flatten_tensors(const std::vector<torch::Tensor>& tensors) {
    return torch::nn::utils::parameters_to_vector(tensors);
}
std::vector<torch::Tensor> unflatten_tensors(const torch::Tensor& flattened_params, const std::vector<torch::IntArrayRef>& tensor_shapes, const std::vector<long long> tensor_sizes) {
    auto flat_tensor_parts = flattened_params.split(tensor_sizes);
    std::vector<torch::Tensor> unflattened_tensors;
    for (size_t i = 0; i < tensor_shapes.size(); i++) {
        unflattened_tensors.push_back(torch::reshape(flat_tensor_parts[i], tensor_shapes[i]));
    }
    return unflattened_tensors;
}
