#include "matcher.h"
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

static constexpr bool USE_ADAPTIVE_BRIGHTNESS = false;

cv::Mat adapt_brightness(const cv::Mat &src) {
    cv::Mat gray_frame;
    cv::cvtColor(src, gray_frame, cv::COLOR_RGB2GRAY);

    cv::Ptr<cv::CLAHE> clahe = createCLAHE(40.0, cv::Size(5, 5));
    clahe->setClipLimit(1);

    cv::Mat clahe_frame;
    clahe->apply(gray_frame, clahe_frame);

    return clahe_frame;
}

int main() {
    cv::Mat src1;
    cv::Mat src2;
    cv::VideoCapture cap("sample_mpg.avi");
    assert(cap.isOpened());

    bool stop = false;
    double rate = cap.get(cv::CAP_PROP_FPS);

    int delay = static_cast<int>(1000.0 / rate);
    std::cout << "Frame rate: " << rate << "\n";

    FeatureMatcher<cv::xfeatures2d::SURF> featureMatcher1{};
    FeatureMatcher<cv::SIFT> featureMatcher3{};
    FeatureMatcher<cv::BRISK> featureMatcher2{};

    while (!stop) {
        bool result = cap.grab();
        if (result) {
            cap >> src1;
            if (USE_ADAPTIVE_BRIGHTNESS) {
                src2 = adapt_brightness(src1);
                src2.copyTo(src1);
            } else {
                src1.copyTo(src2);
            }
        } else {
            std::terminate();
        }

        cv::Mat image_matches1 = featureMatcher1.drawMatch(src1, src2);
        cv::imshow("SURF", image_matches1);

        cv::Mat image_matches2 = featureMatcher2.drawMatch(src1, src2);
        cv::imshow("SIFT", image_matches2);

        cv::Mat image_matches3 = featureMatcher3.drawMatch(src1, src2);
        cv::imshow("BRISK", image_matches3);

        int key = cv::waitKey(delay);
        if (key == 27) {
            stop = true;
        }
    }
    return 0;
}