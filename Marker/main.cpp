#include <cassert>
#include <cstdlib>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

int main() {
    cv::Mat image;
    image = imread("test_marker.jpg", cv::IMREAD_COLOR);

    cv::Mat image_original;
    image.copyTo(image_original);

    // Преобразование в RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // Цветовая фильтрация по красному цвету
    {
        cv::Mat image_;
        cv::cvtColor(image, image_, cv::COLOR_RGB2HSV);
        cv::Mat merged;
        cv::Mat mask1;
        cv::Mat mask2;
        cv::inRange(image_, cv::Scalar(0, 150, 100), cv::Scalar(5, 255, 255), mask1);
        cv::inRange(image_, cv::Scalar(175, 150, 100), cv::Scalar(180, 255, 255), mask2);
        cv::bitwise_and(image_, image_, merged, mask1 | mask2);
        cv::cvtColor(merged, image_, cv::COLOR_HSV2RGB);
        image = image_;
    }

    // Выделение красного квадрата
    {
        cv::Mat image_;
        auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::dilate(image, image_, kernel, cv::Point(-1, -1), 10);
        cv::erode(image_, image_, kernel, cv::Point(-1, -1), 10);

        cv::Mat edges;
        cv::Canny(image_, edges, 255, 150);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        std::vector<std::vector<cv::Point>> hulls;
        for (auto &contour: contours) {
            std::vector<cv::Point> hull;
            cv::convexHull(contour, hull, true);
            hulls.emplace_back(hull);
        }
        cv::drawContours(image_, hulls, -1, cv::Scalar(0, 255, 0));

        assert(hulls.size() == 1);
        auto rect = cv::boundingRect(hulls[0]);
        image = image_original(rect);
    }


    // Выделение черных квадратов
    {
        cv::Mat image_;
        image.copyTo(image_);

        auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::dilate(image_, image_, kernel, cv::Point(-1, -1), 2);
        cv::erode(image_, image_, kernel, cv::Point(-1, -1), 8);
        cv::dilate(image_, image_, kernel, cv::Point(-1, -1), 4);

        cv::cvtColor(image_, image_, cv::COLOR_RGB2HSV);
        cv::inRange(image_, cv::Scalar(0, 0, 0), cv::Scalar(180, 255, 80), image_);

        cv::Mat edges;
        cv::Canny(image_, edges, 140, 40);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        std::vector<std::vector<cv::Point>> hulls;
        for (auto &contour: contours) {
            std::vector<cv::Point> hull;
            cv::convexHull(contour, hull, true);
            hulls.emplace_back(hull);
        }

        std::sort(hulls.begin(), hulls.end(), [](auto &x, auto &y){return x.size() > y.size();});
        hulls.resize(3);

        for (size_t i = 0; i < hulls.size(); ++i) {
            auto rect = cv::boundingRect(hulls[i]);
            cv::Mat res_image = image(rect);
            cv::imshow("Result image " + std::to_string(i), res_image);
        }
    }

    cv::waitKey(0);
    return 0;
}