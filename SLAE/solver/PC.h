#ifndef SOLVER_PC_H
#define SOLVER_PC_H

#include <vector>
#include "Eigen/Dense"

namespace Solver {
    class PC {
    public:
        using Vec = Eigen::ArrayXd;
        using F = std::function<Vec(const Vec&, double)>;

    public:
        PC(F f, Vec vec, double time, double step) : f_(std::move(f)), values_(std::move(vec)), time_(time), step_(step) {}

    public:
        void calc_step() {
            Eigen::VectorXd predictor = values_ + f_(values_, time_) * step_;
            Eigen::VectorXd corrector = values_ + (f_(values_, time_) + f_(predictor, time_ + step_)) * step_ / 2;
            values_ = corrector;
            time_ += step_;
        }

        [[nodiscard]] double get_time() const {
            return time_;
        }

        [[nodiscard]] Vec get_values() const {
            return values_;
        }

    private:
        F f_;
        Vec values_;
        double time_;
        const double step_;
    };
} // Solver



#endif //SOLVER_PC_H
