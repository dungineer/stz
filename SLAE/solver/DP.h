#ifndef SOLVER_DP_H
#define SOLVER_DP_H

#include <iostream>
#include <algorithm>

namespace Solver {
constexpr double MAX_DIFF    = 10e-4;
constexpr double MIN_DIFF    = 10e-6;

class DP {
public:
    using Vec = Eigen::VectorXd;
    using F = std::function<Vec(const Vec&, double)>;

public:
    DP(F f, Vec vec, double time, double step) : f_(std::move(f)), values_(std::move(vec)), time_(time), step_(step) {
        init_butcher();
    }

    void calc_step() {
        Eigen::VectorXd x1;
        Eigen::VectorXd x2;
        Eigen::MatrixXd tmp;
        double diff;
        do {
            tmp = CreateTmpMat();

            x1 = (B[0] * tmp).transpose();
            x2 = (B[1] * tmp).transpose();

            diff = (x1 - x2).cwiseAbs().maxCoeff();
            if (diff > MAX_DIFF) {
                step_ /= 2;
            } else if (diff < MIN_DIFF) {
                step_ *= 2;
            }
            if (step_ > 0.5) {
                step_ = 0.5;
            } else if (step_ < 10e-5) {
                step_ = 10e-5;
            }
        } while (diff > MAX_DIFF);

        values_ += x1 * step_;
        time_ += step_;
    }

    [[nodiscard]] double get_time() const {
        return time_;
    }

    [[nodiscard]] Vec get_values() const {
        return values_;
    }

private:
    Eigen::MatrixXd CreateTmpMat() {
        Eigen::MatrixXd tmp(A.cols(), values_.size());
        Eigen::VectorXd val = values_;
        tmp.row(0) = f_(val, time_);

        for (int i = 1; i < tmp.rows(); ++i) {
            val = values_;
            for (int j = 0; j < i; ++j) {
                val += tmp.row(j) * A(i, j) * step_;
            }
            tmp.row(i) = f_(val, time_ + STEPS(i) * step_);
        }

        return tmp;
    }

    void init_butcher() {
        Eigen::MatrixXd butcher_mat(9, 8);
        butcher_mat << 0, 0, 0, 0, 0, 0, 0, 0, 1.0 / 5, 1.0 / 5, 0, 0, 0, 0, 0, 0, 3.0 / 10, 3.0 / 40, 9.0 / 40, 0, 0, 0, 0,
                0, 4.0 / 5, 44.0 / 45, -56.0 / 15, 32.0 / 9, 0, 0, 0, 0, 8.0 / 9, 19372.0 / 6561, -25360.0 / 2187,
                64448.0 / 6561, -212.0 / 729, 0, 0, 0, 1, 9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176,
                -5103.0 / 18656, 0, 0, 1, 35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0, 0, 35.0 / 384,
                0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0, 0, 5179.0 / 57600, 0, 7571.0 / 16695, 393.0 / 640,
                -92097.0 / 339200, 187.0 / 2100, 1.0 / 40;
        B     = {butcher_mat.block(butcher_mat.rows() - 2, 1, 1, butcher_mat.cols() - 1),
                 butcher_mat.block(butcher_mat.rows() - 1, 1, 1, butcher_mat.cols() - 1)};
        A     = butcher_mat.block(0, 1, butcher_mat.cols() - 1, butcher_mat.cols() - 1);
        STEPS = butcher_mat.block(0, 0, butcher_mat.cols() - 1, 1);
    }

private:
    F f_;
    Vec values_;
    double time_;
    double step_;
    Eigen::VectorXd STEPS;
    std::vector<Eigen::MatrixXd> B;
    Eigen::MatrixXd A;
};
} // Solver


#endif //SOLVER_DP_H
