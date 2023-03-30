#pragma once
#include "ProxyEnv.h"
#include "Box.h"
#include <cmath>
#include <tuple>
//SOURCE: https://gitlab.engr.illinois.edu/jundayu2/nsa-nfm/-/blob/master/src/rllab/rllab/envs/normalized_env.py
class NormalizedEnv: protected ProxyEnv {
    public:
        NormalizedEnv(Env* env, const float& scale_reward = 1, bool normalize_obs = false, bool normalize_reward = false, double obs_alpha = 0.001, double reward_alpha = 0.001);
        //obs should be tensor to match flatten
        void update_obs_estimate(const torch::Tensor& obs);
        void update_reward_estimate(double reward);
        torch::Tensor apply_normalize_obs(const torch::Tensor& obs);
        double apply_normalize_reward(double reward);
        torch::Tensor reset(const torch::Tensor &state = torch::empty(0));
        torch::Tensor get_state() {return wrapped_env_->get_state();}

        //for serialization:
        //def __getstate__
        //def __setstate__


        Space* action_space();

        //TODO: need to add special box behavior to step
        Env::Step step(const torch::Tensor& action);
        Env::Step step_denormalized(const torch::Tensor& denormalized_action);
        torch::Tensor normalize_action(const torch::Tensor& action);
        torch::Tensor denormalize_action(const torch::Tensor& action);

        //originally named sanitize_action
        std::tuple<torch::Tensor, torch::Tensor> normalized_sanitize_action(const torch::Tensor& act);

         



    private:
        float scale_reward_;
        bool normalize_obs_;
        bool normalize_reward_;
        double obs_alpha_;
        double reward_alpha_;
        torch::Tensor obs_mean_;
        torch::Tensor obs_var_;
        float reward_mean_ = 0;
        float reward_var_ = 1;
        
};