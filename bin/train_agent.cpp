#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include "../include/utec/algebra/tensor.h"
#include "../include/utec/nn/neural_network.h"
#include "../include/utec/nn/dense.h"
#include "../include/utec/nn/activation.h"
#include "../include/utec/nn/loss.h"
#include "../include/utec/nn/optimizer.h"
#include "../include/utec/agent/EnvGym.h"

using namespace utec::nn;

int main() {
    const std::string weights_path = "weights.txt";

    auto net = std::make_unique<NeuralNetwork<float>>();
    net->add_layer(std::make_unique<Dense<float>>(3, 8));
    net->add_layer(std::make_unique<ReLU<float>>());
    net->add_layer(std::make_unique<Dense<float>>(8, 1));
    net->add_layer(std::make_unique<Sigmoid<float>>());

    std::cout << "Entrenando modelo en entorno Pong...\n";

    EnvGym env;
    std::vector<std::vector<float>> X_data;
    std::vector<float> Y_data;

    int episodes = 1000;
    const int max_steps = 200;
    std::cout << "Recolectando datos de entrenamiento...\n";
    for (int ep = 0; ep < episodes; ++ep) {
        State s = env.reset();
        bool done = false;
        float reward;
        int steps = 0;

        while (!done && steps++ < max_steps) {
            int action;
            if (s.ball_y > s.paddle_y + 0.02f) action = +1;
            else if (s.ball_y < s.paddle_y - 0.02f) action = -1;
            else action = 0;

            X_data.push_back({s.ball_x, s.ball_y, s.paddle_y});
            float label;
            if (action == -1) label = 0.0f;
            else if (action == 0) label = 0.5f;
            else label = 1.0f;

            Y_data.push_back(label);
            s = env.step(action, reward, done);
        }

        if (ep % 50 == 0)
            std::cout << "Episodio " << ep << " completado, total datos: " << X_data.size() << "\n";
    }

    std::cout << "Recolección completada. Entrenando...\n";

    Tensor<float, 2> X(X_data.size(), 3);
    Tensor<float, 2> Y(Y_data.size(), 1);
    for (size_t i = 0; i < X_data.size(); ++i) {
        for (int j = 0; j < 3; ++j)
            X(i, j) = X_data[i][j];
        Y(i, 0) = Y_data[i];
    }

    const size_t epochs = 300;
    const size_t batch_size = 32;
    const float lr = 0.01f;
    const float decay = 0.001f;

    std::cout << "\nPredicciones antes del entrenamiento:\n";
    auto preds_before = net->predict(X);
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "Input: {" << X(i,0) << ", " << X(i,1) << ", " << X(i,2) << "} -> ";
        std::cout << "Pred: " << preds_before(i,0) << " | Target: " << Y(i,0) << "\n";
    }

    net->train<MSELoss, SGD>(X, Y, epochs, batch_size, lr, decay);

    std::cout << "\nPredicciones después del entrenamiento:\n";
    auto preds_after = net->predict(X);
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "Input: {" << X(i,0) << ", " << X(i,1) << ", " << X(i,2) << "} -> ";
        std::cout << "Pred: " << preds_after(i,0) << " | Target: " << Y(i,0) << "\n";
    }

    net->save_weights(weights_path);
    std::cout << "Modelo entrenado y pesos guardados en: " << weights_path << "\n";

    return 0;
}









