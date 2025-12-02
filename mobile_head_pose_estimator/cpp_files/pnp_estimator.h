#ifndef PNP_ESTIMATOR_H
#define PNP_ESTIMATOR_H

#include "head_pose_estimator.h"

class PnPEstimator : public HeadPoseEstimator {
public:
    std::map<std::string, cv::Mat> estimate(const std::map<std::string, cv::Point2d>& landmarks, const cv::Size& image_size) override;
};

#endif
