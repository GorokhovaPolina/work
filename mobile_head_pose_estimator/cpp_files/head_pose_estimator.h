
#ifndef HEAD_POSE_ESTIMATOR_H
#define HEAD_POSE_ESTIMATOR_H

#include <opencv2/opencv.hpp>

class HeadPoseEstimator {
public:
    virtual ~HeadPoseEstimator() = default;
    virtual std::map<std::string, cv::Mat> estimate(const std::map<std::string, cv::Point2d>& landmarks, const cv::Size& image_size) = 0;
};

#endif
