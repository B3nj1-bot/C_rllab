#include "NormalizedEnv.h"
NormalizedEnv::NormalizedEnv(Env* env, const float& scale_reward, bool normalize_obs, bool normalize_reward, double obs_alpha, double reward_alpha): ProxyEnv(env), scale_reward_(scale_reward), normalize_obs_(normalize_obs), normalize_reward_(normalize_reward), obs_alpha_(obs_alpha), reward_alpha_(reward_alpha)  {
    obs_mean_ = torch::zeros((*(*env).observation_space()).flat_dim());
    obs_var_ = torch::ones((*(*env).observation_space()).flat_dim());
};

void NormalizedEnv::update_obs_estimate(const torch::Tensor& obs) {
    torch::Tensor flat_obs = wrapped_env_->observation_space()->flatten(obs);
    obs_mean_ = (1 - obs_alpha_) * obs_mean_ + obs_alpha_ * flat_obs;
    obs_var_ = (1 - obs_alpha_) * obs_var_ + obs_alpha_ * torch::square(flat_obs - obs_mean_); 

}
void NormalizedEnv::update_reward_estimate(double reward) {
    reward_mean_ = (1 - reward_alpha_) * reward_mean_ + reward_alpha_ * reward;
    reward_var_ = (1 - reward_alpha_) * reward_var_ + reward_alpha_ * pow(reward - reward_mean_, 2);

}
torch::Tensor NormalizedEnv::apply_normalize_obs(const torch::Tensor& obs) {
    update_obs_estimate(obs);
    return ((obs - obs_mean_) / (torch::sqrt(obs_var_) + 1e-8));
}
double NormalizedEnv::apply_normalize_reward(double reward) {
    update_reward_estimate(reward);
    return (reward / (sqrt(reward_var_) + 1e-8));
}
torch::Tensor NormalizedEnv::reset(const torch::Tensor &state) {
    torch::Tensor ret = wrapped_env_->reset(state);
    if (normalize_obs_) {
        return (apply_normalize_obs(ret));
    } else {
        return ret;
    }
}

Space* NormalizedEnv::action_space() {
    if (dynamic_cast<const Box*>(wrapped_env_->action_space()) != nullptr) {
        
        //upper bound
        torch::Tensor ub = torch::ones(wrapped_env_->action_space()->shape());
        static Box box = Box(-1 * ub, ub);
        return &box;
    }
    return wrapped_env_->action_space();
}

//for serialization:
//def __getstate__
//def __setstate__

Env::Step NormalizedEnv::step(const torch::Tensor& action) {
    torch::Tensor scaled_action;
    if (dynamic_cast<const Box*>(wrapped_env_->action_space()) != nullptr) {
        //lower bound
        torch::Tensor lb = wrapped_env_->action_space()->lower_bound();
        torch::Tensor ub = wrapped_env_->action_space()->upper_bound();
        scaled_action = lb + (action + 1.0) * 0.5 * (ub - lb);
        scaled_action = torch::clip(scaled_action, lb, ub);
    } else {
        scaled_action = action;
    }
    Env::Step wrapped_step = wrapped_env_->step(scaled_action);
    torch::Tensor next_obs = wrapped_step.observation;
    double reward = wrapped_step.reward;
    if (normalize_obs_) {
        torch::Tensor next_obs = apply_normalize_obs(wrapped_step.observation);
    }
    if (normalize_reward_) {
        double reward = apply_normalize_reward(wrapped_step.reward);
    }
    return Env::Step(next_obs, reward * scale_reward_, wrapped_step.done, wrapped_step.info);
}
Env::Step NormalizedEnv::step_denormalized(const torch::Tensor& denormalized_action) {
    Env::Step wrapped_step = wrapped_env_->step(denormalized_action);
    torch::Tensor next_obs = wrapped_step.observation;
    double reward = wrapped_step.reward;
    if (normalize_obs_) {
        next_obs = apply_normalize_obs(wrapped_step.observation);
    }
    if (normalize_reward_) {
        reward = apply_normalize_reward(wrapped_step.reward);
    }
    return Env::Step(next_obs, reward * scale_reward_, wrapped_step.done, wrapped_step.info);

}
torch::Tensor NormalizedEnv::normalize_action(const torch::Tensor& action) {
    torch::Tensor scaled_action;
    if (dynamic_cast<const Box*>(wrapped_env_->action_space()) != nullptr) {
        torch::Tensor lb = wrapped_env_->action_space()->lower_bound();
        torch::Tensor ub = wrapped_env_->action_space()->upper_bound();
        torch::Tensor a = action_space()->lower_bound();
        torch::Tensor b = action_space()->upper_bound();
        scaled_action = a + (b - a) * (action - lb) / (ub - lb);
        return torch::clip(scaled_action, a, b);
    }
    return action;
}
torch::Tensor NormalizedEnv::denormalize_action(const torch::Tensor& action) {
     torch::Tensor scaled_action;
    if (dynamic_cast<const Box*>(wrapped_env_->action_space()) != nullptr) {
        torch::Tensor lb = wrapped_env_->action_space()->lower_bound();
        torch::Tensor ub = wrapped_env_->action_space()->upper_bound();
        scaled_action = lb + (action + 1.0) * 0.5 * (ub - lb);
        return torch::clip(scaled_action, lb, ub);
    }
    return action;
}

std::tuple<torch::Tensor, torch::Tensor> NormalizedEnv::normalized_sanitize_action(const torch::Tensor& act) {
    torch::Tensor action = denormalize_action(act);
    torch::Tensor sanitized_action = wrapped_env_->sanitize_action(action);
    return std::make_tuple(normalize_action(sanitized_action), sanitized_action);
}