#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H

#include "../algebra/tensor.h"
#include <random>

namespace utec::utils {

    template <typename T>
    void generate_pong_dataset(utec::algebra::Tensor<T, 2>& X, utec::algebra::Tensor<T, 2>& Y, size_t samples) {
        X = utec::algebra::Tensor<T, 2>({samples, 3});
        Y = utec::algebra::Tensor<T, 2>({samples, 1});

        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<T> dist(0.0, 1.0);

        for (size_t i = 0; i < samples; ++i) {
            T ball_x = dist(rng);
            T ball_y = dist(rng);
            T paddle_y = dist(rng);

            X(i, 0) = ball_x;
            X(i, 1) = ball_y;
            X(i, 2) = paddle_y;


            T dy = ball_y - paddle_y;
            if (std::abs(dy) < static_cast<T>(0.05))
                Y(i, 0) = static_cast<T>(0);
            else if (dy > 0)
                Y(i, 0) = static_cast<T>(+1);
            else
                Y(i, 0) = static_cast<T>(-1);
        }
    }

} // namespace utec::utils

#endif // DATA_GENERATOR_H
