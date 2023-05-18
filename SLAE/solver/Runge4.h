#ifndef CALIBRATION_RUNGE4_H
#define CALIBRATION_RUNGE4_H

#include <vector>
#include "Eigen/Dense"



namespace Solver {
    class Runge4 {
    public:
        using Vec = Eigen::VectorXd;
        using F = std::function<Vec(const Vec&, double)>;

    public:
        Runge4(F f, Vec vec, double time, double step) : f_(std::move(f)), values_(std::move(vec)), time_(time), step_(step) {}

    public:
        void calc_step() {
            Vec k1 = f_(values_, time_);
            Vec k2 = f_(values_ + step_ * k1 / 2, time_ + step_ / 2);
            Vec k3 = f_(values_ + step_ * k2 / 2, time_ + step_ / 2);
            Vec k4 = f_(values_ + step_ * k3, time_ + step_);
            values_ += (k1 + 2 * k2 + 2 * k3 + k4) * step_ / 6;
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
        double step_;
    };
} // Solver



#endif //CALIBRATION_RUNGE4_H
