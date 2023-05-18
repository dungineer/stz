#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>

#include "Runge4.h"
#include "DP.h"
#include "PC.h"

static constexpr double A = 2.5;
static constexpr double B = 1.3;
static constexpr double D = 0.5;
static constexpr double C = D + A / (B * B);

static constexpr double TIME = 5;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "No step argument provided\n";
        return 1;
    }
    double step = std::stod(argv[1]);

    {
        std::cout << "---------- Задача 1 ----------\n";
        Solver::Runge4::F f = [](const Eigen::VectorXd &values, double time){
            Solver::Runge4::Vec res = values;
            res(0) = A * time - B * values(0);
            return res;
        };

        Eigen::VectorXd init_values(1, 1);
        init_values << D;

        Solver::Runge4 solver_runge(f, init_values, 0, step);
        Solver::DP solver_dp(f, init_values, 0, step);

        auto analytical_function = [](double time){
            return (A / B) * (time - 1 / B) + C * std::exp(-B * time);
        };

        double max_diff = 0;
        size_t cnt = 0;
        while (solver_runge.get_time() < TIME) {
            ++cnt;
            double value = solver_runge.get_values()(0);
            double analytical_value = analytical_function(solver_runge.get_time());
            double diff = std::abs(analytical_value - value);
            max_diff = std::max(max_diff, diff);
            solver_runge.calc_step();
        }
        std::cout << "Максимальное рассогласование Runge: " << max_diff << "\n";
        std::cout << "Число шагов Runge: " << cnt << "\n";

        max_diff = 0;
        cnt = 0;
        while (solver_dp.get_time() < TIME) {
            ++cnt;
            double value = solver_dp.get_values()(0);
            double analytical_value = analytical_function(solver_dp.get_time());
            double diff = std::abs(analytical_value - value);
            max_diff = std::max(max_diff, diff);
            solver_dp.calc_step();
        }
        std::cout << "Максимальное рассогласование DP: " << max_diff << "\n";
        std::cout << "Число шагов DP: " << cnt << "\n";
    }

    std::cout << "---------- Задача 2 ----------\n";
    {
        auto f = [](const Eigen::VectorXd &val, double time) {
            Eigen::VectorXd ret(2, 1);
            ret(0, 0) = 9 * val(0, 0) + 24 * val(1, 0) + 5 * cos(time) - 1.0 / 3 * sin(time);
            ret(1, 0) = -24 * val(0, 0) - 51 * val(1, 0) + 9 * cos(time) + 1.0 / 3 * sin(time);
            return ret;
        };

        auto analytical_function1 = [](double time) {
            return 2.0 * exp(-3 * time) - exp(-39 * time) + 1.0 / 3 * cos(time);
        };

        auto analytical_function2 = [](double time) {
            return -exp(-3 * time) + 2.0 * exp(-39 * time) - 1.0 / 3 * cos(time);
        };

        Eigen::VectorXd init_values(2, 1);
        init_values(0, 0) = 4.0 / 3;
        init_values(1, 0) = 2.0 / 3;

        Solver::Runge4 solver_runge(f, init_values, 0, step);
        Solver::DP solver_dp(f, init_values, 0, step);

        double max_diff = 0;
        size_t cnt = 0;
        while (solver_runge.get_time() < TIME) {
            ++cnt;
            double value1 = solver_runge.get_values()(0);
            double value2 = solver_runge.get_values()(1);

            double analytical_value1 = analytical_function1(solver_runge.get_time());
            double analytical_value2 = analytical_function2(solver_runge.get_time());


            double diff1 = std::abs(analytical_value1 - value1);
            double diff2 = std::abs(analytical_value2 - value2);

            max_diff = std::max({max_diff, diff1, diff2});

            solver_runge.calc_step();
        }

        std::cout << "Максимальное рассогласование Runge: " << max_diff << "\n";
        std::cout << "Число шагов Runge: " << cnt << "\n";

        max_diff = 0;
        cnt = 0;
        while (solver_dp.get_time() < TIME) {
            ++cnt;
            double value1 = solver_dp.get_values()(0);
            double value2 = solver_dp.get_values()(1);

            double analytical_value1 = analytical_function1(solver_dp.get_time());
            double analytical_value2 = analytical_function2(solver_dp.get_time());


            double diff1 = std::abs(analytical_value1 - value1);
            double diff2 = std::abs(analytical_value2 - value2);

            max_diff = std::max({max_diff, diff1, diff2});

            solver_dp.calc_step();
        }

        std::cout << "Максимальное рассогласование DP: " << max_diff << "\n";
        std::cout << "Число шагов DP: " << cnt << "\n";
    }
    return 0;
}