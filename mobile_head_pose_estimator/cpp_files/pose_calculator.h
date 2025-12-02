#ifndef POSE_CALCULATOR_H
#define POSE_CALCULATOR_H

#include <opencv2/opencv.hpp>
#include <map>

class GeometricPoseCalculator {
public:
    std::map<std::string, double> calculate_pose(const std::map<std::string, std::any>& data, const std::string& mode = "pnp");
};

#endif
