#include <iostream>
#include <cmath>
#include <fstream>

#include "Eigen/Dense"


using Mat = std::vector<std::vector<double>>;

// Z = 10^(K * T + C)
class Model {
public:
    explicit Model(double k, double c) : k_(k), c_(c) {}

    double operator()(double t) const {
        return std::pow(10, k_ * t + c_);
    }

private:
    double k_;
    double c_;
};

Mat readFile(const std::string &str) {
    Mat res{};

    std::fstream file(str, std::ios::in);
    std::pair<double, double> line;

    while (file.is_open() && !file.eof()) {
        file >> line.first;
        file >> line.second;
        res.push_back({line.first, line.second});
    }

    return res;
}

std::pair<double, double> evaluate(const Mat &data) {
    Eigen::Matrix<double, 12, 2> G;

    for (int i = 0; i < data.size(); ++i) {
        G(i, 0) = data[i][1];
        G(i, 1) = 1;
    }

    Eigen::Vector<double, 12> y;
    for (int i = 0; i < data.size(); ++i) {
        y(i) = std::log10(data[i][0]);
    }

    auto solved = (G.transpose() * G).lu().solve(G.transpose() * y);
    return {solved.matrix()(0, 0), solved.matrix()(1, 0)};
}

Mat filter_ransac(Mat data) {
    static constexpr double THRESHOLD = 5;

    auto count_inliners = [&](std::pair<double, double> hypothesis){
        int c = 0;
        auto model = Model(hypothesis.first, hypothesis.second);
        for (auto &sample : data) {
            if (std::abs(sample[0] - model(sample[1])) < THRESHOLD) {
                ++c;
            }
        }
        return c;
    };

    int max = 0;
    std::pair<double, double> chosen_hypothesis;

    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data.size(); ++j) {
            Mat chosen_pair{data[i], data[j]};
            auto hypothesis = evaluate(chosen_pair);
            int count = count_inliners(hypothesis);
            if (count > max) {
                max = count;
                chosen_hypothesis = hypothesis;
            }
        }
    }

    Mat res;
    auto model = Model(chosen_hypothesis.first, chosen_hypothesis.second);
    for (auto &sample : data) {
        if (std::abs(sample[0] - model(sample[1])) < THRESHOLD) {
            res.push_back(std::move(sample));
        }
    }

    return res;
}

int main() {
    auto data = filter_ransac(readFile("data.txt"));

    auto solved = evaluate(data);
    std::cout << "solved: " << solved.first << " " << solved.second << "\n";

    return 0;
}