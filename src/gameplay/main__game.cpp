#include <iostream>
#include <thread>
#include <conio.h>
#include <chrono>
#include <fstream>
#include "../../include/utec/agent/EnvGym.h"
#include "../../include/utec/agent/PongAgent.h"
#include "../../include/utec/nn/neural_network.h"
#include "../../include/utec/nn/dense.h"
#include "../../include/utec/nn/activation.h"
#include "../utec/ui/display.cpp"

using namespace utec::nn;

int main()
{
    const std::string weights_path = "../utec/agent/weights.txt";

    auto net = std::make_unique<NeuralNetwork<float>>();
    net->add_layer(std::make_unique<Dense<float>>(3, 8));
    net->add_layer(std::make_unique<ReLU<float>>());
    net->add_layer(std::make_unique<Dense<float>>(8, 1));
    net->add_layer(std::make_unique<Sigmoid<float>>());

    std::ifstream fin(weights_path);
    if (fin.good())
    {
        net->load_weights(weights_path);
        std::cout << "Pesos cargados desde: " << weights_path << "\n";
    }
    else
    {
        std::cerr << "No se encontraron pesos. Corre el entrenamiento primero.\n";
        return 1;
    }

    PongAgent<float> agent(std::move(net));
    EnvGym env;

    std::cout << "Bienvenido a Pong IA! Usa W y S para mover tu paleta.\n";
    std::cout << "Presiona cualquier tecla para comenzar...\n";
    std::cin.get();

    int player_score = 0;
    int ai_score = 0;

    while (true)
    {
        auto state = env.reset();
        bool done = false;
        float reward = 0.f;
        int player_action = 0;

        while (!done)
        {
            if (_kbhit())
            {
                char key = _getch();
                if (key == 'w' || key == 'W')
                    player_action = -1;
                else if (key == 's' || key == 'S')
                    player_action = +1;
                else
                    player_action = 0;
            }
            else
            {
                player_action = 0;
            }

            int ai_action = agent.act({state.ball_x, state.ball_y, state.enemy_y});

            state = env.step(player_action, reward, done, ai_action);

            render(state, player_score, ai_score);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (reward > 0)
            player_score++;
        else
            ai_score++;

        std::cout << "\nMarcador -> Tu: " << player_score << " | IA: " << ai_score << "\n";
        std::cout << "Presiona Q para salir, otra tecla para continuar...\n";
        char c = _getch();
        if (c == 'q' || c == 'Q')
            break;
    }

    std::cout << "\nGracias por jugar!\n";
    return 0;
}
