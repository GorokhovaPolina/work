#include "pnp_estimator.h"

std::map<std::string, cv::Mat> PnPEstimator::estimate(const std::map<std::string, cv::Point2d>& landmarks, const cv::Size& image_size) {
    int w = image_size.width;
    int h = image_size.height;

    cv::Mat model_points = (cv::Mat_<float>(4, 3) <<
        0.0f, 0.0f, 0.0f,
        -0.065f, 0.035f, -0.03f,
        0.065f, 0.035f, -0.03f,
        0.0f, -0.065f, -0.04f);

    cv::Mat image_points = (cv::Mat_<float>(4, 2) <<
        landmarks.at("nose").x, landmarks.at("nose").y,
        landmarks.at("left_eye").x, landmarks.at("left_eye").y,
        landmarks.at("right_eye").x, landmarks.at("right_eye").y,
        landmarks.at("mouth").x, landmarks.at("mouth").y);

    cv::Mat K = (cv::Mat_<float>(3, 3) << w, 0, w / 2.0f, 0, h, h / 2.0f, 0, 0, 1);
    cv::Mat dist = cv::Mat::zeros(4, 1, CV_32F);

    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(model_points, image_points, K, dist, rvec, tvec, false, cv::SOLVEPNP_EPNP);

    if (!success) {
        return {};
    }

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6;

    double yaw, pitch, roll;
    if (!singular) {
        yaw = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
        pitch = std::atan2(-R.at<double>(2, 0), sy);
        roll = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
    }
    else {
        yaw = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        pitch = std::atan2(-R.at<double>(2, 0), sy);
        roll = 0;
    }

    std::map<std::string, cv::Mat> result;
    result["yaw"] = cv::Mat(1, 1, CV_64F, std::degrees(yaw));
    result["pitch"] = cv::Mat(1, 1, CV_64F, std::degrees(pitch));
    result["roll"] = cv::Mat(1, 1, CV_64F, std::degrees(roll));
    result["rvec"] = rvec;
    result["tvec"] = tvec;
    result["K"] = K;
    result["dist"] = dist;

    return result;
}
