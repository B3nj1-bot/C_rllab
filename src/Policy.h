#pragma once
#include "EnvSpec.h"
#include "Parameterized.h"
#include "Placeholder.h"
//SOURCE: https://gitlab.engr.illinois.edu/jundayu2/nsa-nfm/-/blob/master/src/rllab/rllab/policies/base.py
class Policy: protected Parameterized {
    public:
        Policy(EnvSpec env_spec): env_spec_(env_spec) {};

        virtual torch::Tensor get_action(const torch::Tensor& observation);
        virtual void reset();
        Space* observation_space() { return env_spec_.observation_space(); }
        Space* action_space() { return env_spec_.action_space(); }
        //Indicates whether the policy is recurrent
        bool recurrent() {return false; }
        //Placeholder should be the type of paths
        void log_diagnostics(const std::vector<Placeholder>& paths) { return; }
        std::vector<std::string> state_info_keys() {return std::vector<std::string>(); }
        // "Clean up operation," not sure where this is actually implemented
        void terminate() {return;}
    protected:
        EnvSpec env_spec_;

        
    
        
        
};