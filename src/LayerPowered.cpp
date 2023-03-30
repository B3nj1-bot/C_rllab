#include "LayerPowered.h"

torch::nn::ParameterList LayerPowered::get_params_internal(const std::map<std::string, bool>& tags) {
    bool regularizable;
    bool trainable;
    if (tags.count("regularizable") > 0) {
        regularizable = tags.at("regularizable");
    }
    if (tags.count("trainable") > 0) {
        trainable = tags.at("trainable");
    }
    torch::nn::ParameterList parameters;
    for (size_t i = 0; i < output_layers_->size(); i++) {
        //p is dictionary with "weight" or "bias" as key and tensor w/ parameter as value
        for (auto p: output_layers_[i]->named_parameters()) {
            if (trainable && !p.value().requires_grad()) {
                continue;
            }
            if (regularizable && p.key() == "bias") {
                continue;
            }
            parameters->append(p);
        }
        
    }
    return parameters;
};