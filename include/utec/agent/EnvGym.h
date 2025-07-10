#ifndef ENV_GYM_H
#define ENV_GYM_H

#include "State.h"
#include <random>
#include <algorithm>
#include <ctime>

namespace utec::nn {

    class EnvGym {
        State state;
        float enemy_y;
        std::default_random_engine rng;
        float ball_dx, ball_dy;
        float player_speed;
        float ai_speed;
        int steps;
        int max_steps;

    public:
        EnvGym() :
            ball_dx(0.03f), ball_dy(0.02f),
            player_speed(0.07f),
            ai_speed(0.04f),
            steps(0), max_steps(200)
        {
            rng.seed(static_cast<unsigned>(std::time(nullptr)));
            reset();
        }

        State get_state() const {
            return State{
                state.ball_x,
                state.ball_y,
                state.paddle_y,
                enemy_y
            };
        }

        State reset() {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            std::uniform_real_distribution<float> dir(-0.03f, 0.03f);

            state.ball_x = 0.5f;
            state.ball_y = dist(rng);
            state.paddle_y = dist(rng);
            enemy_y = dist(rng);

            ball_dx = (rng() % 2 == 0 ? 1 : -1) * (0.02f + std::abs(dir(rng)));
            ball_dy = (rng() % 2 == 0 ? 1 : -1) * (0.01f + std::abs(dir(rng)));

            steps = 0;
            return get_state();
        }


        State step(int action, float& reward, bool& done, int ai_action) {
            steps++;


            state.paddle_y += action * player_speed;
            state.paddle_y = std::clamp(state.paddle_y, 0.0f, 1.0f);

            enemy_y += ai_action * ai_speed;
            enemy_y = std::clamp(enemy_y, 0.0f, 1.0f);

            state.ball_x += ball_dx;
            state.ball_y += ball_dy;


            if (state.ball_y <= 0.0f || state.ball_y >= 1.0f)
                ball_dy *= -1;

            done = false;
            reward = 0.0f;


            if (state.ball_x >= 1.0f) {
                if (std::abs(state.ball_y - state.paddle_y) < 0.1f) {
                    ball_dx *= -1;
                    state.ball_x = 0.99f;

                    float dy_jitter = (static_cast<float>(rng()) / rng.max() - 0.5f) * 0.02f;
                    ball_dy += dy_jitter;

                    reward = 1.0f;
                } else {
                    reward = -1.0f;
                    done = true;
                    return reset();
                }
            }


            if (state.ball_x <= 0.0f) {
                if (std::abs(state.ball_y - enemy_y) < 0.1f) {
                    ball_dx *= -1;
                    state.ball_x = 0.01f;

                    float dy_jitter = (static_cast<float>(rng()) / rng.max() - 0.5f) * 0.02f;
                    ball_dy += dy_jitter;

                    // recompensa negativa porque la IA lo bloqueó
                    reward = -0.2f;
                } else {
                    reward = +1.0f;  // el jugador ganó punto
                    done = true;
                    return reset();
                }
            }

            if (steps >= max_steps) {
                done = true;
            }

            return get_state();
        }
    };

} // namespace utec::nn

#endif // ENV_GYM_H
