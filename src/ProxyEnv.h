#pragma once
#include "Env.h"
#include "Placeholder.h"

//SOURCE:: https://gitlab.engr.illinois.edu/jundayu2/nsa-nfm/-/blob/master/src/rllab/rllab/envs/proxy_env.py
class ProxyEnv: protected Env {
    public:
        ProxyEnv(Env* wrapped_env): wrapped_env_(wrapped_env) {};
        //Placeholder will be the type of an "observation"
        //should technically use **kwargs
        torch::Tensor reset(const torch::Tensor &state = torch::empty(0)) {return wrapped_env_->reset(state);}
        Space* action_space() { return wrapped_env_->action_space();}
        Space* observation_space() {return wrapped_env_->observation_space();}

        Env::Step step(const torch::Tensor& action) {return wrapped_env_->step(action);}

        //Python version
        //def render(self, *args, **kwargs):
        //return self._wrapped_env.render(*args, **kwargs)
        void render();
        //Python version
        // def log_diagnostics(self, paths, *args, **kwargs):
        // self._wrapped_env.log_diagnostics(paths, *args, **kwargs)
        void log_diagnostics(const std::vector<Placeholder>& paths) {return wrapped_env_->log_diagnostics(paths);}
        //not sure what this returns. Used in GymEnv
        //i.e. self._horizon = env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']
        Placeholder horizon() { return wrapped_env_->horizon();}
        //"clean up operation"
        void terminate() {wrapped_env_->terminate();}

        //no idea where these are used
        Placeholder get_param_values() { return wrapped_env_->get_param_values(); }
        void set_param_values(Placeholder params) { wrapped_env_->set_param_values(params); }

    protected:
        Env* wrapped_env_;
        
        
};