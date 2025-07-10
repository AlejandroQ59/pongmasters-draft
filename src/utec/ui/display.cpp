#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include "../../../include/utec/agent/State.h"

const int WIDTH = 40;
const int HEIGHT = 20;

void render(const State &state, int player_score, int ai_score)
{
    std::vector<std::string> screen(HEIGHT + 1, std::string(WIDTH, ' '));

    std::string score_line = " IA: " + std::to_string(ai_score) + "   Jugador: " + std::to_string(player_score);
    for (size_t i = 0; i < score_line.size() && i < WIDTH; ++i)
        screen[0][i] = score_line[i];

    for (int x = 0; x < WIDTH; ++x) {
        screen[1][x] = '.';
        screen[HEIGHT][x] = '.';
    }

    for (int y = 1; y < HEIGHT; ++y) {
        screen[y][0] = '.';
        screen[y][WIDTH - 1] = '.';
    }

    int ball_x = static_cast<int>(state.ball_x * (WIDTH - 2)) + 1;
    int ball_y = static_cast<int>(state.ball_y * (HEIGHT - 2)) + 1;
    int paddle_y = static_cast<int>(state.paddle_y * (HEIGHT - 4)) + 1;
    int enemy_y = static_cast<int>(state.enemy_y * (HEIGHT - 4)) + 1;

    if (ball_y > 1 && ball_y < HEIGHT && ball_x > 0 && ball_x < WIDTH - 1)
        screen[ball_y][ball_x] = 'O';

    for (int i = 0; i < 3; ++i) {
        int y = paddle_y + i;
        if (y > 1 && y < HEIGHT) {
            screen[y][WIDTH - 3] = '[';
            screen[y][WIDTH - 2] = '[';
        }
    }

    for (int i = 0; i < 3; ++i) {
        int y = enemy_y + i;
        if (y > 1 && y < HEIGHT) {
            screen[y][1] = ']';
            screen[y][2] = ']';
        }
    }

#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif

    for (const auto &line : screen)
        std::cout << line << '\n';
}
