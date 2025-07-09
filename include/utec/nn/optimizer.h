//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H

#include <unordered_map>
#include <cmath>
#include "interfaces.h"

namespace utec::nn {

template<typename T>
class SGD final : public IOptimizer<T> {
    T learning_rate;
    T momentum;
    std::unordered_map<Tensor<T,2>*, Tensor<T,2>> velocity;
    
public:
    explicit SGD(T learning_rate = 0.01, T momentum = 0.9)
        : learning_rate(learning_rate), momentum(momentum) {}
        
    void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
        
        if (velocity.find(&params) == velocity.end()) {
            velocity[&params] = Tensor<T,2>(params.shape()[0], params.shape()[1]);
            velocity[&params].fill(0);
        }
        
        auto& v = velocity[&params];
        
        for(size_t i = 0; i < params.shape()[0]; ++i) {
            for(size_t j = 0; j < params.shape()[1]; ++j) {
                v(i,j) = momentum * v(i,j) + learning_rate * grads(i,j);
                params(i,j) -= v(i,j);
            }
        }
    }
    
    void set_learning_rate(T new_lr) { learning_rate = new_lr; }
};

template<typename T>
class Adam final : public IOptimizer<T> {
    T learning_rate;
    T beta1, beta2, epsilon;
    std::size_t t = 0;
    std::unordered_map<Tensor<T,2>*, std::pair<Tensor<T,2>, Tensor<T,2>>> states;
    
public:
    explicit Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
        : learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon) {}
        
    void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {

        if (states.find(&params) == states.end()) {
            auto& state = states[&params];
            state.first = Tensor<T,2>(params.shape()[0], params.shape()[1]);
            state.second = Tensor<T,2>(params.shape()[0], params.shape()[1]);
            state.first.fill(0);
            state.second.fill(0);
        }
        
        auto& [m, v] = states[&params];
        t++;
        
        for (size_t i = 0; i < params.shape()[0]; ++i) {
            for (size_t j = 0; j < params.shape()[1]; ++j) {
                m(i,j) = beta1 * m(i,j) + (1 - beta1) * grads(i,j);
                v(i,j) = beta2 * v(i,j) + (1 - beta2) * grads(i,j) * grads(i,j);
                
                T m_hat = m(i,j) / (1 - std::pow(beta1, t));
                T v_hat = v(i,j) / (1 - std::pow(beta2, t));
                
                params(i,j) -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            }
        }
    }
    
    void step() override {
        
    }
    
    void set_learning_rate(T new_lr) { learning_rate = new_lr; }
};
}
#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H