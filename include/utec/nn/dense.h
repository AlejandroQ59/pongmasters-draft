#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include "interfaces.h"
#include "optimizer.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdexcept>

namespace utec::nn {
  template<typename T>
  class Dense final : public ILayer<T> {
  public:
    std::size_t in_f, out_f;
    Tensor<T,2> last_x;
    Tensor<T,2> last_z;
    Tensor<T,2> w, b, dw, db;

    template<typename InitWFun, typename InitBFun>
    Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun)
      : in_f(in_f), out_f(out_f), w(in_f, out_f), b(1, out_f), dw(in_f, out_f), db(1, out_f)
    {
      init_w_fun(w);
      init_b_fun(b);
    }

    Dense(size_t in_f, size_t out_f)
    : Dense(in_f, out_f,
        [](Tensor<T, 2>& t) {
            for (auto& v : t.data_)
              v = static_cast<T>(rand()) / RAND_MAX;
      },
      [](Tensor<T, 2>& t) {
          for (auto& v : t.data_)
            v = T(0);
    }) {}


    Tensor<T,2> forward(const Tensor<T,2>& x) override {
      this->last_x = x;
      this->last_z = x * w + b;
      return last_z;
    }

    Tensor<T,2> backward(const Tensor<T,2>& dZ) override {
      Tensor<T,2> dZ_mod = dZ;

      if (out_f == 1) {
        for (size_t i = 0; i < last_z.data_.size(); ++i) {
          T sigmoid_val = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-last_z.data_[i]));
          dZ_mod.data_[i] = dZ.data_[i] * sigmoid_val * (1 - sigmoid_val);
        }
      }

      auto dx = dZ_mod * transpose_2d(w);
      dw = transpose_2d(last_x) * dZ_mod;

      db.fill(0);
      for (size_t i = 0; i < dZ.shape()[0]; ++i) {
        for (size_t j = 0; j < out_f; ++j) {
          db(0, j) += dZ_mod(i, j);
        }
      }

      return dx;
    }

    void update_params(IOptimizer<T>& optimizer) override {
      optimizer.update(w, dw);
      optimizer.update(b, db);
    }

    // === Guardar y cargar pesos ===
    void save_weights(std::ostream& os) const {
      for (const auto& val : w.data_) os << val << " ";
      os << "\n";
      for (const auto& val : b.data_) os << val << " ";
      os << "\n";
    }

    void load_weights(std::istream& is) {
      for (auto& val : w.data_) is >> val;
      for (auto& val : b.data_) is >> val;
    }
  };
}

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
