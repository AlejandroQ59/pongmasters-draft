#ifndef PONG_AGENT_H
#define PONG_AGENT_H

#include "../nn/neural_network.h"
#include "../algebra/tensor.h"
#include <memory>
#include <cmath>
#include "State.h"

namespace utec::nn {

    template <typename T>
    class PongAgent {
        std::unique_ptr<NeuralNetwork<T>> model;

    public:
        PongAgent() = default;

        PongAgent(std::unique_ptr<NeuralNetwork<T>> m)
            : model(std::move(m)) {}

        int act(const State& s) {
            Tensor<T, 2> input(1, 3);
            input(0, 0) = static_cast<T>(s.ball_x);
            input(0, 1) = static_cast<T>(s.ball_y);
            input(0, 2) = static_cast<T>(s.paddle_y);

            auto output = model->predict(input);
            T decision = output(0, 0);

            if (decision < static_cast<T>(0.45)) {
                return -1;
            } else if (decision > static_cast<T>(0.55)) {
                return +1;
            } else {
                return 0;
            }
        }



    };

}

#endif



