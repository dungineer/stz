#ifndef LIST_GAUSS_H
#define LIST_GAUSS_H


#include <utility>
#include <algorithm>
#include <stack>


class Gauss {
public:
    static constexpr double SINGULARITY_THRESHOLD = 0.000001;

    using Mat = std::vector<std::vector<double>>;
    using Vec = std::vector<double>;

    Gauss(const Mat &a, const Vec &b) {
        a_ = a;
        a_i_ = a;
        fill_identity(a_i_);
        b_ = b;
        solve();
    }

    Vec get_solution() {
        return b_;
    }

    Mat get_a_inversed() {
        return a_i_;
    }

private:
    void solve() {
        for (int i = 0; i < a_.size() - 1; ++i) {
            choose_max_row(i);
            if (std::abs(a_[i][i]) < SINGULARITY_THRESHOLD) {
                throw std::runtime_error("Singular matrix");
            }

            for (int j = i + 1; j < a_.size(); ++j) {
                double m = a_[j][i] / a_[i][i];
                for (int k = 0; k < a_.size(); ++k) {
                    a_[j][k] -= a_[i][k] * m;
                    a_i_[j][k] -= a_i_[i][k] * m;
                }
                b_[j] -= b_[i] * m;
            }
        }
        for (int i = static_cast<int>(a_.size()) - 1; i >= 0; --i) {
            for (int k = 1; k < a_.size() - i; ++k) {
                b_[i] -= a_[i][i + k] * b_[i + k];
            }
            b_[i] /= a_[i][i];
        }
        for (int i = static_cast<int>(a_.size()) - 1; i >= 0; --i) {
            for (int j = i - 1; j >= 0; --j) {
                double m = a_[j][i] / a_[i][i];
                for (int k = static_cast<int>(a_.size()) - 1; k >= 0; --k) {
                    a_i_[j][k] -= a_i_[i][k] * m;
                }
            }
            std::for_each(a_i_[i].begin(), a_i_[i].end(), [&](auto &val){ val /= a_[i][i]; });
        }
    }

    void choose_max_row(int i) {
        int max_i = i;
        double max = std::abs(a_[i][i]);
        for (int j = i + 1; j < a_.size(); ++j) {
            if (std::abs(a_[j][i]) > max) {
                max = std::abs(a_[j][i]);
                max_i = j;
            }
        }
        a_[i].swap(a_[max_i]);
        a_i_[i].swap(a_i_[max_i]);
        std::swap(b_[i], b_[max_i]);
    }

    static void fill_identity(Mat &a) {
        for (int i = 0; i < a.size(); ++i) {
            for (int j = 0; j < a.size(); ++j) {
                a[i][j] = (i == j? 1 : 0);
            }
        }
    }

private:
    Mat a_;
    Mat a_i_;
    Vec b_;
};


#endif // LIST_GAUSS_H
