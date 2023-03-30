#pragma once
#include "EnvSpec.h"
//SOURCE: https://gitlab.engr.illinois.edu/jundayu2/nsa-nfm/-/blob/master/src/rllab/rllab/policies/base.py
class Policy {
    public:
        Policy(EnvSpec env_spec): env_spec_(env_spec) {};

        virtual torch::Tensor get_action(const torch::Tensor& observation);
    protected:
        EnvSpec env_spec_;

        
    
        
        
};