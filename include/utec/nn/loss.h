//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#include "interfaces.h"
#include <cmath>

namespace utec::nn{
 
template<typename T>
class MSELoss final: public ILoss<T, 2> {
public:
  Tensor<T,2> y_prediction, y_true;

  MSELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_true) {
    this->y_prediction = y_prediction;
    this->y_true = y_true;
  }

  T loss() const override {
    T diff = 0;
    for(size_t i = 0; i < y_prediction.data_.size(); ++i) {
      T error = y_prediction.data_[i] - y_true.data_[i];
      diff += error*error;
    }
    return diff/y_prediction.data_.size();
  }

  Tensor<T,2> loss_gradient() const override {
    Tensor<T,2> res(y_prediction.shape_);
    for(size_t i = 0; i<y_prediction.data_.size(); ++i) {
      res.data_[i] = 2*(y_prediction.data_[i] - y_true.data_[i])/y_prediction.data_.size();
    }
    return res;
   }
};


template<typename T>
class BCELoss final: public ILoss<T, 2> {
public:
  Tensor<T,2> y_prediction, y_true;
  BCELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_true) {
    this->y_prediction = y_prediction;
    this->y_true = y_true;
  }
  T loss() const override {
    // formulon = 1/n (sum y_true_i * log(y_prediction_i) + (1-y_true_i) * log(1-y_prediction_i))
    T res = 0;
    for(size_t i = 0; i<y_true.data_.size(); ++i) {
      T p = std::max(std::min(y_prediction.data_[i], T(0.999)), T(0.001)); // para evitar log(0)
      res += y_true.data_[i] * std::log(p) + (1-y_true.data_[i]) * std::log(1-p);
    }
    return -res/y_true.data_.size();
  }

  Tensor<T,2> loss_gradient() const override {
    Tensor<T,2> res(y_true.shape_);
    for(size_t i = 0; i<y_true.data_.size(); ++i){
      res.data_[i] = -((y_true.data_[i]/y_prediction.data_[i]) - ((1-y_true.data_[i])/(1-y_prediction.data_[i])))/y_true.data_.size();
    }
    return res;
  }
}; 
}
#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
