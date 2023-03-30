#pragma once
#include "Policy.h"
#include "LayerPowered.h"
class DeterministicMLPPolicy: protected Policy, protected LayerPowered {
    public:
        DeterministicMLPPolicy(EnvSpec env_spec, const std::vector<int> hidden_sizes = std::vector<int>({32, 32}), std::string hidden_nonlinearity = "rectify", std::string hidden_W_init = "HeUniform", std::tuple<std::string, double> hidden_b_init = std::make_tuple("Constant", 0.), std::string output_nonlinearity = "tanh", std::tuple<std::string, double, double> output_W_init = std::make_tuple("Uniform", -0.003, 0.003), std::tuple<std::string, double, double> output_b_init = std::make_tuple("Uniform", -0.003, 0.003), bool bn = false);
    
        
        
};