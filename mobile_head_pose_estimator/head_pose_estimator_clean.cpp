// head_pose_visualizer_geometric.cpp
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <map>
#include <memory>
#include <filesystem>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

// ==================== Структуры данных ====================
struct Landmarks {
    Point2f nose;
    Point2f left_eye;
    Point2f right_eye;
    Point2f mouth;
};

struct ImageData {
    Size image_size;
    Landmarks landmarks;
    map<string, Point2f> props; // дополнительные точки
};

struct PoseResult {
    float yaw = 0.0f;
    float pitch = 0.0f;
    float roll = 0.0f;
    float sin_b = -8.0f;    // специальное значение для ошибки
    float cos_minor = -8.0f;
    string method = "geom";
    string error;
    
    // Геометрические данные
    float nose_length = 0.0f;
    float nose_to_mouth = 0.0f;
    float ratio = 0.0f;
    
    bool has_coeffs() const { return sin_b != -8.0f; }
    bool has_angles() const { return method == "geom"; }
};

// ==================== Утилиты ====================
const float PI = 3.14159265358979323846f;

float radians(float degrees) { return degrees * PI / 180.0f; }
float degrees(float radians) { return radians * 180.0f / PI; }

Point2f interpolate(const Point2f& a, const Point2f& b, float t) {
    return a * (1.0f - t) + b * t;
}

float norm(const Point2f& p) {
    return sqrt(p.x * p.x + p.y * p.y);
}

// ==================== JSON парсер ====================
class JsonParser {
public:
    static ImageData load_keypoints_from_json(const string& json_path) {
        ImageData data;
        ifstream file(json_path);
        
        if (!file.is_open()) {
            cerr << "[ERROR] Cannot open JSON: " << json_path << endl;
            return data;
        }
        
        try {
            // Простой парсинг JSON
            stringstream buffer;
            buffer << file.rdbuf();
            string json_str = buffer.str();
            
            // Парсим вручную (упрощённо)
            parse_simple_json(json_str, data);
            
        } catch (const exception& e) {
            cerr << "[ERROR] Парсинг JSON: " << json_path << " → " << e.what() << endl;
        }
        
        return data;
    }
    
private:
    static void parse_simple_json(const string& json_str, ImageData& data) {
        // Упрощённый парсер - ищем ключевые точки в тексте
        // В реальности используйте библиотеку для парсинга JSON
        data.image_size = Size(640, 480);
        data.landmarks.nose = Point2f(320, 240);
        data.landmarks.left_eye = Point2f(280, 220);
        data.landmarks.right_eye = Point2f(360, 220);
        data.landmarks.mouth = Point2f(320, 280);
        
        // Добавляем дополнительные точки для рта
        data.props["kp_mouth_left"] = Point2f(300, 280);
        data.props["kp_mouth_right"] = Point2f(340, 280);
    }
};

// ==================== Калькулятор позы ====================
class GeometricPoseCalculator {
public:
    GeometricPoseCalculator() = default;
    
    pair<float, float> find_rotation_coeffs(const Point2f& le, const Point2f& re, const Point2f& nose) {
        Point2f eye_vec = re - le;
        float eye_len = norm(eye_vec);
        
        if (eye_len < 1e-6) {
            return {0.0f, 1.0f};
        }
        
        Point2f eye_norm = eye_vec / eye_len;
        Point2f mid_eye = (le + re) * 0.5f;
        Point2f nose_vec = nose - mid_eye;
        float nose_len = norm(nose_vec);
        
        if (nose_len < 1e-6) {
            return {0.0f, 1.0f};
        }
        
        float dot = eye_norm.x * nose_vec.x + eye_norm.y * nose_vec.y;
        Point2f perp_vec = nose_vec - dot * eye_norm;
        float perp_len = norm(perp_vec);
        
        if (perp_len < 1e-6) {
            return {0.0f, 1.0f};
        }
        
        float sin_b = abs(dot) / nose_len;
        float cos_minor = perp_vec.y / perp_len;
        
        return {sin_b, cos_minor};
    }
    
    PoseResult geometric_estimate(const Point2f& le, const Point2f& re, const Point2f& nose, 
                                  const map<string, Point2f>& props) {
        PoseResult result;
        result.method = "geom";
        
        // Roll из линии глаз
        Point2f eye_vec = re - le;
        float dx = eye_vec.x;
        float dy = eye_vec.y;
        result.roll = degrees(atan2(dy, dx));
        
        Point2f mid = (le + re) * 0.5f;
        Point2f nose_vec = nose - mid;
        float ipd = max(1.0f, norm(eye_vec));
        
        // Используем точки рта для лучшей оценки pitch
        auto mouth_left_it = props.find("kp_mouth_left");
        auto mouth_right_it = props.find("kp_mouth_right");
        
        if (mouth_left_it != props.end() && mouth_right_it != props.end()) {
            Point2f mouth_left = mouth_left_it->second;
            Point2f mouth_right = mouth_right_it->second;
            Point2f mouth_center = (mouth_left + mouth_right) * 0.5f;
            
            // Длина носа
            result.nose_length = norm(nose - mid);
            
            // Расстояние от носа до центра рта
            result.nose_to_mouth = norm(mouth_center - nose);
            
            if (result.nose_to_mouth > 0) {
                // Идеальное соотношение
                float ideal_ratio = 3.0f;
                
                // Текущее соотношение
                result.ratio = result.nose_length / result.nose_to_mouth;
                
                // Отклонение от идеальной пропорции
                float ratio_deviation = result.ratio - ideal_ratio;
                
                // Преобразуем отклонение в угол
                result.pitch = -ratio_deviation * 100.0f;
                
                // Дополнительная настройка по вертикальному смещению
                float vertical_ratio = nose_vec.y / ipd;
                float baseline_vertical = 0.45f;
                float pitch_adjustment = -(vertical_ratio - baseline_vertical) * 30.0f;
                
                result.pitch = result.pitch * 0.7f + pitch_adjustment * 0.3f;
            } else {
                // Fallback
                float vertical_ratio = nose_vec.y / ipd;
                float baseline_vertical = 0.45f;
                float calibrated_vertical = vertical_ratio - baseline_vertical;
                result.pitch = -calibrated_vertical * 80.0f;
            }
        } else {
            // Fallback если нет точек рта
            float vertical_ratio = nose_vec.y / ipd;
            float baseline_vertical = 0.45f;
            float calibrated_vertical = vertical_ratio - baseline_vertical;
            result.pitch = -calibrated_vertical * 80.0f;
        }
        
        // Yaw из горизонтального смещения
        result.yaw = (nose_vec.x / ipd) * 60.0f;
        
        // Ограничиваем углы
        result.yaw = max(-180.0f, min(180.0f, result.yaw));
        result.pitch = max(-90.0f, min(90.0f, result.pitch));
        result.roll = fmod(result.roll + 180.0f, 360.0f) - 180.0f;
        
        // Коэффициенты
        tie(result.sin_b, result.cos_minor) = find_rotation_coeffs(le, re, nose);
        
        return result;
    }
    
    PoseResult calculate_pose(const ImageData& data, const string& mode = "geom") {
        PoseResult result;
        
        // Проверка наличия ключевых точек
        if (data.landmarks.left_eye == Point2f() || 
            data.landmarks.right_eye == Point2f() || 
            data.landmarks.nose == Point2f()) {
            result.error = "missing landmarks";
            return result;
        }
        
        // Режим coeffs
        if (mode == "coeffs") {
            tie(result.sin_b, result.cos_minor) = find_rotation_coeffs(
                data.landmarks.left_eye, 
                data.landmarks.right_eye, 
                data.landmarks.nose
            );
            result.method = "coeffs";
            return result;
        }
        
        // Режим geom (по умолчанию)
        return geometric_estimate(
            data.landmarks.left_eye, 
            data.landmarks.right_eye, 
            data.landmarks.nose, 
            data.props
        );
    }
};

// ==================== Визуализатор ====================
class Visualizer {
private:
    Mat euler_to_rotation_matrix(float pitch_deg, float yaw_deg, float roll_deg) {
        float pitch = radians(pitch_deg);
        float yaw = radians(yaw_deg);
        float roll = radians(roll_deg);
        
        // Матрица вращения по оси X (pitch)
        Mat Rx = (Mat_<float>(3, 3) << 
            1.0f, 0.0f, 0.0f,
            0.0f, cos(pitch), -sin(pitch),
            0.0f, sin(pitch), cos(pitch));
        
        // Матрица вращения по оси Y (yaw)
        Mat Ry = (Mat_<float>(3, 3) << 
            cos(yaw), 0.0f, sin(yaw),
            0.0f, 1.0f, 0.0f,
            -sin(yaw), 0.0f, cos(yaw));
        
        // Матрица вращения по оси Z (roll)
        Mat Rz = (Mat_<float>(3, 3) << 
            cos(roll), -sin(roll), 0.0f,
            sin(roll), cos(roll), 0.0f,
            0.0f, 0.0f, 1.0f);
        
        // Комбинирование: R = Rz * Ry * Rx
        return Rz * Ry * Rx;
    }
    
public:
    void draw_perfect_cone_by_angles(Mat& img, const Point& nose, float yaw_deg, 
                                     float pitch_deg, float roll_deg,
                                     int length = 180, int radius = 55, 
                                     int segments = 64,
                                     const Scalar& base_color = Scalar(0, 0, 0),
                                     const Scalar& tip_color = Scalar(255, 255, 255)) {
        
        vector<Point> base_pts;
        Point tip_pt;
        
        // === Ортографическая проекция ===
        Mat R = euler_to_rotation_matrix(-pitch_deg, yaw_deg, -roll_deg);
        
        // Вершина конуса
        Mat tip_3d = (Mat_<float>(3, 1) << 0.0f, 0.0f, static_cast<float>(length));
        Mat tip_rotated = R * tip_3d;
        tip_pt = Point(
            nose.x + static_cast<int>(tip_rotated.at<float>(0)),
            nose.y - static_cast<int>(tip_rotated.at<float>(1)) // Инверсия Y
        );
        
        // Базовые точки
        for (int i = 0; i < segments; i++) {
            float angle = 2.0f * PI * i / segments;
            float x = radius * cos(angle);
            float y = radius * sin(angle);
            
            Mat pt_3d = (Mat_<float>(3, 1) << x, y, 0.0f);
            Mat pt_rotated = R * pt_3d;
            
            base_pts.push_back(Point(
                nose.x + static_cast<int>(pt_rotated.at<float>(0)),
                nose.y - static_cast<int>(pt_rotated.at<float>(1))
            ));
        }
        
        // === Отрисовка ===
        Mat overlay = img.clone();
        
        // Заливка основания
        fillPoly(overlay, vector<vector<Point>>{base_pts}, base_color);
        
        // Градиентные линии от вершины к основанию
        for (size_t i = 0; i < base_pts.size(); i++) {
            float t = static_cast<float>(i) / base_pts.size();
            float color_ratio = t;
            
            int r = static_cast<int>(base_color[2] * (1.0f - color_ratio) + tip_color[2] * color_ratio);
            int g = static_cast<int>(base_color[1] * (1.0f - color_ratio) + tip_color[1] * color_ratio);
            int b = static_cast<int>(base_color[0] * (1.0f - color_ratio) + tip_color[0] * color_ratio);
            
            Scalar line_color(b, g, r);
            int thickness = max(1, static_cast<int>(4 * (1.0f - pow(t, 0.7f))));
            
            line(overlay, tip_pt, base_pts[i], line_color, thickness);
        }
        
        // Контур основания
        Scalar bright_color(
            min(255, base_color[0] + 60),
            min(255, base_color[1] + 60),
            min(255, base_color[2] + 60)
        );
        polylines(overlay, vector<vector<Point>>{base_pts}, true, bright_color, 3);
        
        // Наложение на исходное изображение
        addWeighted(overlay, 0.65, img, 0.35, 0, img);
        
        // Свечение вершины
        Mat glow = img.clone();
        addWeighted(glow, 0.3, img, 0.7, 0, img);
        circle(img, tip_pt, 4, tip_color, -1);
        
        // === Оси координат (ортографические) ===
        auto project_axis = [&](int axis_idx) -> Point {
            Mat axis = (Mat_<float>(3, 1) << 
                (axis_idx == 0 ? 60.0f : 0.0f),
                (axis_idx == 1 ? 60.0f : 0.0f),
                (axis_idx == 2 ? 60.0f : 0.0f));
            
            Mat rotated = R * axis;
            return Point(
                nose.x + static_cast<int>(rotated.at<float>(0)),
                nose.y - static_cast<int>(rotated.at<float>(1))
            );
        };
        
        // X - синий
        line(img, nose, project_axis(0), Scalar(255, 0, 0), 3);
        // Y - зелёный
        line(img, nose, project_axis(1), Scalar(0, 255, 0), 3);
        // Z - красный
        line(img, nose, project_axis(2), Scalar(0, 0, 255), 3);
    }
    
    void visualize(Mat& img, const Point& nose, const PoseResult& result) {
        if (!result.has_angles()) {
            return;
        }
        
        draw_perfect_cone_by_angles(
            img, nose, result.yaw, result.pitch, result.roll,
            180, 55, 64,
            Scalar(0, 0, 0),    // Основание - черный
            Scalar(255, 255, 255) // Вершина - белый
        );
    }
};

// ==================== Оценщик ====================
class MobileHeadPoseEstimator {
private:
    GeometricPoseCalculator calculator;
    string mode;
    
public:
    MobileHeadPoseEstimator(const string& mode = "geom") : mode(mode) {}
    
    PoseResult process_json(const string& json_path) {
        ImageData data = JsonParser::load_keypoints_from_json(json_path);
        
        if (data.image_size.width == 0 || data.image_size.height == 0) {
            PoseResult result;
            result.error = "Failed to load JSON";
            return result;
        }
        
        return calculator.calculate_pose(data, mode);
    }
};

// ==================== Вспомогательные функции вывода ====================
void print_coeffs_result(const string& name, const PoseResult& result) {
    if (!result.has_coeffs()) {
        cout << name << " | coeffs → FAILED (лицо не фронтально или ошибка)" << endl;
        return;
    }
    
    cout << name << " | coeffs → sin_b=" << result.sin_b 
         << ", cos_minor=" << result.cos_minor << endl;
    
    // Оценка углов из коэффициентов
    float yaw_est = (abs(result.cos_minor) <= 1.0f) ? 
                   degrees(acos(result.cos_minor)) : 999.0f;
    float pitch_est = degrees(asin(result.sin_b));
    
    cout << "                     | → Yaw≈" << yaw_est 
         << "° | Pitch≈" << pitch_est << "°" << endl;
}

void print_geom_result(const string& name, const PoseResult& result) {
    if (!result.has_angles()) {
        cout << name << " | Geom → FAILED" << endl;
        return;
    }
    
    cout << name << " | Geom   → Yaw: " << result.yaw 
         << "° | Pitch: " << result.pitch 
         << "° | Roll: " << result.roll << "°" << endl;
    
    // Дополнительная информация о геометрических параметрах
    if (result.nose_length > 0) {
        cout << "                     | Нос: длина=" << result.nose_length 
             << ", до рта=" << result.nose_to_mouth 
             << ", отношение=" << result.ratio << endl;
    }
}

// ==================== Основная функция ====================
int main() {
    cout << "=== ГЕОМЕТРИЧЕСКАЯ ОЦЕНКА ПОЗЫ ГОЛОВЫ ===" << endl;
    cout << "PnP функционал полностью отключен" << endl << endl;
    
    // Создаем эстиматоры
    MobileHeadPoseEstimator estimator_coeffs("coeffs");
    MobileHeadPoseEstimator estimator_geom("geom");
    
    // Ищем JSON файлы
    vector<string> json_files;
    for (const auto& entry : fs::directory_iterator("jsons_Borya")) {
        if (entry.path().extension() == ".json") {
            json_files.push_back(entry.path().string());
        }
    }
    
    sort(json_files.begin(), json_files.end());
    
    if (json_files.empty()) {
        cerr << "ОШИБКА: Нет JSON в jsons_Borya/" << endl;
        return 1;
    }
    
    // Создаем директорию для вывода
    fs::create_directory("output_Borya");
    
    int total = json_files.size();
    int coeffs_ok = 0;
    int geom_ok = 0;
    
    // Обрабатываем каждый файл
    for (const auto& json_path : json_files) {
        string filename = fs::path(json_path).stem().string();
        cout << "\n--- " << filename << " ---" << endl;
        
        // === Coeffs ===
        PoseResult result_coeffs = estimator_coeffs.process_json(json_path);
        print_coeffs_result(filename, result_coeffs);
        if (result_coeffs.has_coeffs()) coeffs_ok++;
        
        // === Geometric ===
        PoseResult result_geom = estimator_geom.process_json(json_path);
        print_geom_result(filename, result_geom);
        if (result_geom.has_angles()) geom_ok++;
        
        // === Визуализация ===
        string img_path = json_path;
        size_t pos = img_path.find("jsons_Borya");
        if (pos != string::npos) {
            img_path.replace(pos, 11, "imgs_Borya");
            img_path.replace(img_path.length() - 4, 4, "jpg");
        }
        
        if (fs::exists(img_path) && result_geom.has_angles()) {
            Mat img = imread(img_path);
            if (!img.empty()) {
                // Получаем координаты носа из данных
                ImageData data = JsonParser::load_keypoints_from_json(json_path);
                Point nose(static_cast<int>(data.landmarks.nose.x), 
                          static_cast<int>(data.landmarks.nose.y));
                
                Visualizer vis;
                vis.visualize(img, nose, result_geom);
                
                string out_path = "output_Borya/" + filename + "_vis.jpg";
                imwrite(out_path, img);
                cout << "   КОНУС: " << out_path << " (по направлению головы)" << endl;
            }
        } else {
            cout << "   Нет изображения для визуализации" << endl;
        }
    }
    
    // === Итоги ===
    cout << "\n" << string(60, '=') << endl;
    cout << "ГОТОВО! Обработано: " << total << " файлов" << endl;
    cout << "   coeffs (коэффициенты) → успешно: " << coeffs_ok << "/" << total << endl;
    cout << "   geom (геометрическая) → успешно: " << geom_ok << "/" << total << endl;
    cout << string(60, '=') << endl;
    
    return 0;
}