#pragma once
#include "LayerPowered.h"
#include "QFunction.h"
#include "EnvSpec.h"
//SOURCE: https://gitlab.engr.illinois.edu/jundayu2_research/nsa-nfm/-/blob/master/src/rllab/rllab/q_functions/continuous_mlp_q_function.py
class ContinuousMLPQFunction: protected QFunction, protected LayerPowered {
    public:
        ContinuousMLPQFunction(EnvSpec env_spec, const std::vector<int> hidden_sizes = std::vector<int>({32, 32}), std::string hidden_nonlinearity = "rectify", std::string hidden_W_init = "HeUniform", std::tuple<std::string, double> hidden_b_init = std::make_tuple("Constant", 0.), int action_merge_layer = -2, std::string output_nonlinearity = "", std::tuple<std::string, double, double, double> output_W_init = std::make_tuple("Uniform", -0.003, 0.003, 0), std::tuple<std::string, double, double, double> output_b_init = std::make_tuple("Uniform", -0.003, 0.003, 0), bool bn = false);
    private:
        torch::nn::Sequential layers;
        
};