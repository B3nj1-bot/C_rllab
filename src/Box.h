#pragma once
//Placeholder class
class Box: public Space {
    //a box in R^n, each coordinate is bounded
    public:
        Box(const torch::Tensor& low, const torch::Tensor& high, std::vector<int64_t> shape = std::vector<int64_t>());
    private:
        int nothing = 0;
        
};