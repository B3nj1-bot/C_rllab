#include <iostream>
#include <torch/torch.h>
#include "DDPG.h"
int main() {
  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;
}