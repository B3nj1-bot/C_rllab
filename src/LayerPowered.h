#pragma once
#include "Parameterized.h"
//SOURCE: https://gitlab.engr.illinois.edu/jundayu2/nsa-nfm/-/blob/master/src/rllab/rllab/core/lasagne_powered.py
class LayerPowered: protected Parameterized {
    public:
        LayerPowered() = default;
        LayerPowered(torch::nn::ModuleList output_layers): output_layers_(output_layers) {};
        torch::nn::ModuleList output_layers() { return output_layers_;}
        torch::nn::ParameterList get_params_internal(const std::map<std::string, bool>& tags = std::map<std::string, bool>());
    protected:
        //Module List, might be Sequential List. A list of modules containining linear output layers
       torch::nn::ModuleList output_layers_;

        
};