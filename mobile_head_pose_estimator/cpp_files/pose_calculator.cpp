#include "pose_calculator.h"

std::pair<double, double> GeometricPoseCalculator::find_rotation_coeffs(const cv::Point2d& le, const cv::Point2d& re, const cv::Point2d& no) {
    cv::Point2d mid = (le + re) / 2.0;
    cv::Point2d v = no - mid;
    double norm = cv::norm(v);
    if (norm < 1e-6) return { -8.0, 0.0 };
    double sin_b = v.y / norm;
    sin_b = std::max(-1.0, std::min(1.0, sin_b));
    double cos_minor = std::sqrt(std::max(0.0, 1.0 - sin_b * sin_b));
    return { sin_b, cos_minor };
}

std::map<std::string, double> GeometricPoseCalculator::geom_estimate(const cv::Point2d& le, const cv::Point2d& re, const cv::Point2d& no) {
    cv::Point2d eye_vec = re - le;
    double dx = eye_vec.x;
    double dy = eye_vec.y;
    double roll = std::degrees(std::atan2(dy, dx));

    cv::Point2d mid = (le + re) / 2.0;
    cv::Point2d nose_vec = no - mid;
    double ipd = std::max(1.0, cv::norm(eye_vec));

    double vertical_ratio = nose_vec.y / ipd;
    double baseline_vertical = 0.3;
    double calibrated_vertical = vertical_ratio - baseline_vertical;
    double pitch = -calibrated_vertical * 60.0;
    double yaw = (nose_vec.x / ipd) * 40.0;

    yaw = std::max(-180.0, std::min(180.0, yaw));
    pitch = std::max(-90.0, std::min(90.0, pitch));
    roll = fmod(roll + 180.0, 360.0) - 180.0;

    auto [sin_b, cos_minor] = find_rotation_coeffs(le, re, no);

    return {
        {"yaw", yaw},
        {"pitch", pitch},
        {"roll", roll},
        {"sin_b", sin_b},
        {"cos_minor", cos_minor},
        {"method", 1.0}  // Code for 'geom'
    };
}

double GeometricPoseCalculator::norm_angle(double a) {
    a = fmod(a + 180.0, 360.0) - 180.0;
    if (std::abs(a) < 1e-6) a = 0.0;
    return a;
}

cv::Mat rotation_matrix_to_euler(const cv::Mat& R) {
    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6;

    double x, y, z;
    if (!singular) {
        x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    }
    else {
        x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return (cv::Mat_<double>(3, 1) << x, y, z);
}

std::map<std::string, double> GeometricPoseCalculator::calculate_pose(const PoseData& data, const std::string& mode) {
    if (data.landmarks.empty()) return { {"error", 1.0} };

    cv::Point2d left_eye = data.landmarks.at("left_eye");
    cv::Point2d right_eye = data.landmarks.at("right_eye");
    cv::Point2d nose = data.landmarks.at("nose");
    int w = data.image_size.width;
    int h = data.image_size.height;

    if (mode == "coeffs") {
        auto [sin_b, cos_minor] = find_rotation_coeffs(left_eye, right_eye, nose);
        if (sin_b == -8.0) return { {"error", 1.0} };
        return { {"sin_b", sin_b}, {"cos_minor", cos_minor} };
    }

    if (mode == "geom") {
        return geom_estimate(left_eye, right_eye, nose);
    }

    // PnP mode
    cv::Mat image_points = (cv::Mat_<double>(3, 1) << left_eye.x, left_eye.y, 1.0,
        right_eye.x, right_eye.y, 1.0,
        nose.x, nose.y, 1.0);  // Only 3 points? Original has 4, but in geom 3

    // Adjust for 3 points if model_points is for 3
    cv::Mat model_points = (cv::Mat_<double>(3, 3) << -0.065, 0.035, -0.03,  // left_eye
        0.065, 0.035, -0.03,   // right_eye
        0.0, 0.0, 0.0);        // nose

    double focal_length = static_cast<double>(w);
    cv::Point2d center(w / 2.0, h / 2.0);
    cv::Mat K = (cv::Mat_<double>(3, 3) << focal_length, 0.0, center.x,
        0.0, focal_length, center.y,
        0.0, 0.0, 1.0);
    cv::Mat dist = cv::Mat::zeros(4, 1, CV_64F);

    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(model_points, image_points, K, dist, rvec, tvec, false, cv2::SOLVEPNP_ITERATIVE);

    if (!success) {
        auto geom = geom_estimate(left_eye, right_eye, nose);
        geom["error"] = 1.0;  // Add error flag
        return geom;
    }

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    cv::Mat angles_rad = rotation_matrix_to_euler(R);

    double pitch = std::degrees(angles_rad.at<double>(0));
    double yaw = std::degrees(angles_rad.at<double>(1));
    double roll = std::degrees(angles_rad.at<double>(2));

    return {
        {"yaw", norm_angle(yaw)},
        {"pitch", norm_angle(pitch)},
        {"roll", norm_angle(roll)},
        {"method", 2.0}  // Code for 'pnp'
        // Add rvec, tvec, K, dist if needed, but since map<double>, perhaps separate map for mats
    };
}
