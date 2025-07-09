//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#include "interfaces.h"
#include <cmath>

namespace utec::nn {

 template<typename T>
  class ReLU final : public ILayer<T> {
    Tensor<T,2> last_input;
    public:

    Tensor<T,2> forward(const Tensor<T,2>& z) override {
        last_input = z;
        Tensor<T,2> output(z.shape());
        for(size_t i = 0; i < z.size(); ++i) {
            output.data_[i] = std::max(static_cast<T>(0), z.data_[i]);
        }
        return output;
    }


    Tensor<T,2> backward(const Tensor<T,2>& grad) override {
        Tensor<T,2> output_grad(grad.shape());
        for(size_t i = 0; i < grad.size(); ++i) {
            output_grad.data_[i] = grad.data_[i] * 
                (last_input.data_[i] > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0));
        }
        return output_grad;
    }
  };

  template<typename T>
  class Sigmoid final : public ILayer<T> {
    Tensor<T,2> output;
    public:
    Tensor<T,2> forward(const Tensor<T,2>& z) override {
        output.reshape(z.shape_);
        // formulon: 1 / (1+e^-x)
        for(size_t i = 0; i<z.size(); ++i){
           if(z.data_[i] >= 0) {
                T exp_val = std::exp(-z.data_[i]);
                output.data_[i] = static_cast<T>(1) / (static_cast<T>(1) + exp_val);
            } else {
                T exp_val = std::exp(z.data_[i]);
                output.data_[i] = exp_val / (static_cast<T>(1) + exp_val);
            }
        }
        return output;
    }

    Tensor<T,2> backward(const Tensor<T,2>& g) override {
         Tensor<T,2> output_grad(g.shape_);
         //formulon: (1 / (1+e^-x)) * (1 - (1 / (1+e^-x)))
         const double e = std::exp(1.0);
         for(size_t i = 0; i < g.size(); ++i) {
            output_grad.data_[i] = g.data_[i] * output.data_[i] * (static_cast<T>(1) - output.data_[i]);
        }
        return output_grad;
    }
  };
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
