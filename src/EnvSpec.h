#pragma once
#include "Space.h"
//Source: https://gitlab.engr.illinois.edu/jundayu2/nsa-nfm/-/blob/master/src/rllab/rllab/envs/env_spec.py
class EnvSpec  {
    public:
        EnvSpec(Space *observation_space, Space *action_space): observation_space_(observation_space), action_space_(action_space) {};
        Space* observation_space() {
            return observation_space_;
        }
        Space* action_space() {
            return action_space_;
        }
    private:
        Space* observation_space_;
        Space* action_space_;
    
        
        
};