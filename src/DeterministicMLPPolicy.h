#pragma once
#include "Policy.h"
#include "LayerPowered.h"
#include <cmath>
//SOURCE: https://gitlab.engr.illinois.edu/jundayu2_research/nsa-nfm/-/blob/master/src/rllab/rllab/policies/deterministic_mlp_policy.py
class DeterministicMLPPolicy: protected Policy, protected LayerPowered, torch::nn::Module {
    public:
        DeterministicMLPPolicy(EnvSpec env_spec, const std::vector<int> hidden_sizes = std::vector<int>({32, 32}), std::string hidden_nonlinearity = "rectify", std::string hidden_W_init = "HeUniform", std::tuple<std::string, double> hidden_b_init = std::make_tuple("Constant", 0.), std::string output_nonlinearity = "tanh", std::tuple<std::string, double, double, double> output_W_init = std::make_tuple("Uniform", -0.003, 0.003, 0), std::tuple<std::string, double, double, double> output_b_init = std::make_tuple("Uniform", -0.003, 0.003, 0), bool bn = false);
        torch::Tensor get_action(const torch::Tensor& observation);
        torch::Tensor get_actions(const torch::Tensor& observations);

        //no equivalent, just use layers->forward()
        //def get_action_sym(self, obs_var):
        //return L.get_output(self._output_layer, obs_var)


    private:
        torch::nn::Sequential layers;
    
       
};