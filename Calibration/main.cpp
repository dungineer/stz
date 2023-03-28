#include <iostream>
#include <cmath>
#include "Eigen/Dense"


int main() {
    Eigen::Matrix<double, 12, 2> G{
        {71, 1},
        {64, 1},
        {52, 1},
        {41, 1},
        {33, 1},
        {23, 1},
        {17, 1},
        {12, 1},
        {2, 1},
        {0, 1},
        {87, 1},
        {-5, 1}
    };

    Eigen::Vector<double, 12> y{27, 31, 43, 58, 69, 86, 102, 111, 122, 137, 18, 176};

    for (int i = 0; i < y.rows(); ++i) {
        y(i) = std::log10(y(i));
    }
    std::cout << "y:\n" << y << "\n";

    std::cout << "G:\n" << G << "\n";

    auto b = G.transpose() * y;
    std::cout << "b:\n" << b << "\n";

    std::cout << "LU solution:\n" << (G.transpose() * G).lu().solve(b) << "\n";
    auto QR = G.colPivHouseholderQr().solve(y);
    std::cout << "QR solution:\n" << QR << "\n";

    return 0;
}