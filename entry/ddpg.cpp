#include <iostream>
#include <torch/torch.h>
#include "tensor_utils.h"
#include "DDPG.h"


int main() {
  torch::Tensor tensor1 = torch::ones({3, 4});
  torch::Tensor tensor2 = torch::ones({2, 2});
  torch::Tensor tensor3 = torch::ones({5, 5});
  torch::Tensor flattened = flatten_tensors(std::vector<torch::Tensor>({tensor1, tensor2, tensor3}));
  std::vector<torch::Tensor> unflattened_flattened = unflatten_tensors(flattened, std::vector<torch::IntArrayRef>({{3, 4}, {2,2}, {5, 5}}), std::vector<long long>({12, 4, 25}));
  std::cout << tensor1 << '\n';
  std::cout << tensor2 << '\n';
  std::cout << tensor3 << '\n';
  std::cout << flattened << '\n';
  for (auto t: unflattened_flattened) {
    std::cout << t << '\n';
  }
  
}