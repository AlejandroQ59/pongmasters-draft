#include "../include/utec/agent/EnvGym.h"
#include "../include/utec/agent/PongAgent.h"
#include "../include/utec/nn/neural_network.h"
#include "../include/utec/nn/dense.h"
#include "../include/utec/nn/activation.h"
#include <memory>
#include <iostream>
#include <fstream>

using namespace utec::nn;

int main() {
    const std::string weights_path = "weights.txt";

    // Crear y cargar red neuronal
    auto net = std::make_unique<NeuralNetwork<float>>();
    net->add_layer(std::make_unique<Dense<float>>(3, 8));
    net->add_layer(std::make_unique<ReLU<float>>());
    net->add_layer(std::make_unique<Dense<float>>(8, 1));
    net->add_layer(std::make_unique<Sigmoid<float>>());

    std::ifstream fin(weights_path);
    if (fin.good()) {
        net->load_weights(weights_path);
        std::cout << "Pesos cargados desde: " << weights_path << "\n";
    } else {
        std::cerr << "No se encontraron pesos. Asegúrate de entrenar antes.\n";
        return 1;
    }

    PongAgent<float> agent(std::move(net));
    EnvGym env;

    int wins = 0;
    int total_episodes = 1000;
    int min_bounce_win = 3;

    for (int ep = 0; ep < total_episodes; ++ep) {
        auto state = env.reset();
        bool done = false;
        float reward = 0.f;
        float total_reward = 0.f;

        while (!done) {
            int action = agent.act(state);
            state = env.step(action, reward, done);
            total_reward += reward;
        }

        if (total_reward >= min_bounce_win) {
            wins++;
            std::cout << "Episodio " << ep << ": Recompensa total = " << total_reward << " [GANO]\n";
        } else {
            std::cout << "Episodio " << ep << ": Recompensa total = " << total_reward << " [PERDIO]\n";
        }
    }

    float win_rate = 100.f * wins / total_episodes;
    std::cout << "\nResumen final: Gano " << wins << " de " << total_episodes << " episodios. (Win rate: " << win_rate << "%)\n";
    std::cout << "\n==============================================\n";
    std::cout << "  Nota: Un episodio se considera ganado si el\n";
    std::cout << "  agente logra al menos " << min_bounce_win << " rebotes exitosos.\n";
    std::cout << "==============================================\n";


    return 0;
}







