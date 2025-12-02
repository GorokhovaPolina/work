#include "json_parser.h"

KeypointsData load_keypoints_from_json(const std::string& json_path) {
    try {
        std::ifstream f(json_path);
        nlohmann::json data = nlohmann::json::parse(f);

        int w = data["image_size"][0];
        int h = data["image_size"][1];
        nlohmann::json props = data["props"];

        cv::Point2d left_inner(props["kp_eye_left_inner"][0], props["kp_eye_left_inner"][1]);
        cv::Point2d left_outer(props["kp_eye_left_outer"][0], props["kp_eye_left_outer"][1]);
        cv::Point2d right_inner(props["kp_eye_right_inner"][0], props["kp_eye_right_inner"][1]);
        cv::Point2d right_outer(props["kp_eye_right_outer"][0], props["kp_eye_right_outer"][1]);

        cv::Point2d left_eye = (left_inner + left_outer) / 2.0;
        cv::Point2d right_eye = (right_inner + right_outer) / 2.0;

        cv::Point2d mouth_left(props["kp_mouth_left"][0], props["kp_mouth_left"][1]);
        cv::Point2d mouth_right(props["kp_mouth_right"][0], props["kp_mouth_right"][1]);
        cv::Point2d mouth = (mouth_left + mouth_right) / 2.0;

        cv::Point2d nose(props["kp_nose_tip"][0], props["kp_nose_tip"][1]);

        std::map<std::string, cv::Point2d> lm = {
            {"nose", nose},
            {"left_eye", left_eye},
            {"right_eye", right_eye},
            {"mouth", mouth}
        };

        return { cv::Size(w, h), lm };
    }
    catch (std::exception& e) {
        std::cout << "[ERROR] Парсинг JSON: " << json_path << " → " << e.what() << std::endl;
        return { cv::Size(0, 0), {} };
    }
}
