#ifndef DETECT_MATCHER_H
#define DETECT_MATCHER_H

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

template<class DetectorType>
class FeatureMatcher {
public:
    explicit FeatureMatcher() {
        detector_ = DetectorType::create();
        matcher_ = cv::BFMatcher::create();
    }

    cv::Mat drawMatch(const cv::Mat &src1, const cv::Mat &src2) {
        std::vector<cv::KeyPoint> keys1;
        std::vector<cv::KeyPoint> keys2;

        cv::Mat descr1;
        cv::Mat descr2;
        cv::Mat image_matches;

        detector_->detectAndCompute(src1, cv::noArray(), keys1, descr1);
        detector_->detectAndCompute(src2, cv::noArray(), keys2, descr2);

        std::vector<cv::DMatch> matches;
        matcher_->match(descr1, descr2, matches);
        cv::drawMatches(src1, keys1, src2, keys2, matches, image_matches);

        return image_matches;
    }
private:
    cv::Ptr<cv::FeatureDetector> detector_;
    cv::Ptr<cv::BFMatcher> matcher_;
};


#endif //DETECT_MATCHER_H
