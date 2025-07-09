#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <conio.h> // Para _kbhit() y _getch() en Windows
#include "../../include/utec/agent/EnvGym.h"

using namespace utec;

const int WIDTH = 40;
const int HEIGHT = 20;

const char BALL_CHAR = 'O';
const char PADDLE_CHAR = '|';
const char EMPTY_CHAR = ' ';

void clear_screen() {
    std::system("cls");
}

// Dibuja el campo
void draw(const State& state) {
    int ball_x = static_cast<int>(state.ball_x * WIDTH);
    int ball_y = static_cast<int>(state.ball_y * HEIGHT);
    int paddle_y = static_cast<int>(state.paddle_y * HEIGHT);
    int enemy_y = static_cast<int>(state.enemy_y * HEIGHT);

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            if (x == 0 || x == WIDTH - 1) std::cout << '#'; // Bordes
            else if (x == 2 && std::abs(y - paddle_y) <= 1) std::cout << PADDLE_CHAR;
            else if (x == WIDTH - 3 && std::abs(y - enemy_y) <= 1) std::cout << PADDLE_CHAR;
            else if (x == ball_x && y == ball_y) std::cout << BALL_CHAR;
            else std::cout << EMPTY_CHAR;
        }
        std::cout << '\n';
    }
}

// Maneja entrada del usuario (teclas W/S)
int player_input() {
    if (_kbhit()) {
        char key = _getch();
        if (key == 'w' || key == 'W') return -1;
        if (key == 's' || key == 'S') return 1;
    }
    return 0;
}

// Bucle principal de juego
void run_game(agent::EnvGym& env, PongAgent<float>& agent) {
    float reward = 0.f;
    bool done = false;
    auto state = env.reset();

    while (!done) {
        clear_screen();
        draw(state);

        int human_action = player_input(); // Teclas W/S
        int ai_action = agent.act(state);  // -1, 0, 1

        state = env.step(human_action, reward, done, ai_action);

        std::this_thread::sleep_for(std::chrono::milliseconds(80));
    }

    std::cout << "\n¡Juego terminado!\n";
}
