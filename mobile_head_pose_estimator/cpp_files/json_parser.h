#ifndef JSON_PARSER_H
#define JSON_PARSER_H

#include <fstream>
#include <map>
#include <opencv2/opencv.hpp>
#include "nlohmann/json.hpp"

struct KeypointsData {
    cv::Size image_size;
    std::map<std::string, cv::Point2d> landmarks;
};

KeypointsData load_keypoints_from_json(const std::string& json_path);

#endif
