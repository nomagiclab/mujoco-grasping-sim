#include <algorithm>
#include <iostream>
#include <cmath>
#include <chrono>

#include <opencv2/imgcodecs.hpp>

#include "mujoco.h"

using namespace cv;
using namespace std;

#define PI 3.14159265358979323846

const char project_path[] = "../myproject/mujoco-grasping-sim/";
const char xmlfile[] = "gripper.xml";

#define WIDTH 10
#define HEIGHT 10

#define debug false


mjtNum get_tm_diff(chrono::steady_clock::time_point tm_start)
{
    chrono::duration<double> elapsed = chrono::steady_clock::now() - tm_start;
    return elapsed.count();
}


// Searches element id through all elements in the model.
// Input : all models object, type of element, name of element in the xml
// Output : Element's id.
// Throws error when element is not found within the xml.
int get_id(const mjModel *m, int object, const char* name) {
    int object_id = mj_name2id(m, object, name);
    if (object_id == -1) {
        throw std::runtime_error("mj_name2id error");
    }
    return object_id;
}


tuple<mjtNum, mjtNum, mjtNum> get_gripper_cam_coords(const mjModel *m, mjData *d) {
    auto cam_body_id = get_id(m, mjOBJ_BODY, "gripper-cam-body");
    return make_tuple(d->xpos[3 * cam_body_id], d->xpos[3 * cam_body_id + 1], d->xpos[3 * cam_body_id + 2]);
}


// Checks whether gripper holds an object.
// Input : All models object, data of where models are located
// Returns true if gripper holds an object, false otherwise.
bool gripper_holds(const mjModel *m, mjData *d) {
    auto left_gripper_id  = get_id(m, mjOBJ_BODY, "arm1_body");
    auto right_gripper_id = get_id(m, mjOBJ_BODY, "arm2_body");
    // Could not find a better solution as of now.
    if (debug)
        cout << d->xpos[3 * left_gripper_id ] << " " << d->xpos[3 * left_gripper_id + 1 ] << " " << d->xpos[3 * left_gripper_id + 2 ] << " | " <<
                d->xpos[3 * right_gripper_id] << " " << d->xpos[3 * right_gripper_id + 1] << " " << d->xpos[3 * right_gripper_id + 2] << "\n";

    auto diff = hypot((d->xpos[3 * left_gripper_id] - d->xpos[3 * right_gripper_id]), (d->xpos[3 * left_gripper_id + 1] - d->xpos[3 * right_gripper_id + 1]));

    cout << "diff = " << diff << "\n";

    return diff > 0.1;
}


pair<mjtNum, mjtNum> get_body_coords(const mjModel *m, pair<int, int> pixel_height_width,
                                     tuple<mjtNum, mjtNum, mjtNum> cam_coords) {
    if (HEIGHT != WIDTH) {
        throw std::runtime_error("get_body_coords error: WIDTH and HEIGHT has to be the same");
    }

    int cam_id = get_id(m, mjOBJ_CAMERA, "gripper-cam");

    mjtNum y_coord_diff = tan((m->cam_fovy[cam_id] / 2) * (PI / 180)) * get<2>(cam_coords) * 2;
    mjtNum x_coord_diff = y_coord_diff;

    mjtNum x_coord = get<0>(cam_coords) + (static_cast<double>(pixel_height_width.second) - static_cast<double>(WIDTH) / 2)
                                          / WIDTH * x_coord_diff;
    mjtNum y_coord = get<1>(cam_coords) - (static_cast<double>(pixel_height_width.first) - static_cast<double>(HEIGHT) / 2)
                                          / HEIGHT * y_coord_diff;

    return make_pair(x_coord, y_coord);
}


void simulate(mjtNum sim_time, const mjModel *m, mjData *d) {
    mjtNum sim_start = d->time;
    while (d->time < sim_start + sim_time) {
        mjtNum frame_sim_start = d->time;
        while (d->time - frame_sim_start < 1.0 / 60.0) {
            mj_step(m, d);
        }
    }
}


void move_to(const mjModel *m, mjData *d, pair <int, int> goal) {

    tuple<mjtNum, mjtNum, mjtNum> cam_coords = get_gripper_cam_coords(m, d);
    pair<mjtNum, mjtNum> closest_coords = get_body_coords(m, goal, cam_coords);

    d->ctrl[0] = closest_coords.first;
    d->ctrl[1] = closest_coords.second;

    simulate(1, m, d);
}


void move_down(const mjModel *m, mjData *d) {
    d->ctrl[2] = -3.5;
    simulate(2, m, d);
}


void move_up(const mjModel *m, mjData *d) {
    d->ctrl[2] = 0;
    simulate(2, m, d);
}


void grab(const mjModel *m, mjData *d) {
    d->ctrl[3] = 0.5;
    d->ctrl[4] = 0.5;
    simulate(2, m, d);
}


void rotate(const mjModel *m, mjData *d, double radian_angle) {
    d->ctrl[5] = radian_angle;
}


void save_success_map(const bool arr[HEIGHT][WIDTH], int map_number){
    //Crete the image base
    Mat image = Mat::zeros(HEIGHT, WIDTH, CV_8U);

    //Check for failure
    if (image.empty()) {
        printf("Could not create the image!!\n");
        return;
    }

    // Put correct values of pixels
    for(int i = 0; i < HEIGHT; ++i){
        for(int j = 0; j < WIDTH; ++j){
            image.at<unsigned char>(i, j) = 255 * arr[i][j];
        }
    }

    string path = "../myproject/mujoco-grasping-sim/success_map" + to_string(map_number) + ".png";
    imwrite(path, image);
}


vector <pair <int, int>> get_attempt_positions() {
    vector <pair <int, int>> positions;

    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            positions.emplace_back(i, j);
        }
    }
    return positions;
}


int main() {
    char xmlpath[100] = {};
    strcat(xmlpath, project_path);
    strcat(xmlpath, xmlfile);

    char error[1000] = "Could not load binary model";
    mjModel *m = mj_loadXML(xmlpath, nullptr, error, 1000);

    mjvCamera gripper_cam;
    gripper_cam.type = mjCAMERA_FIXED;
    gripper_cam.fixedcamid = get_id(m, mjOBJ_CAMERA, "gripper-cam");
    gripper_cam.trackbodyid = -1;
    
    vector <pair <int, int>> attempt_positions = get_attempt_positions();

    for (int map_number = 0; map_number < 8; map_number++) {
        chrono::steady_clock::time_point sim_tm_start = chrono::steady_clock::now();

        bool success_map[HEIGHT][WIDTH];

        for(int i = 0; i < HEIGHT; ++i){
            for(int j = 0; j < WIDTH; ++j){
                success_map[i][j] = false;
            }
        }

        unsigned long all_attempts = attempt_positions.size();

        for (int iter = 0; iter < all_attempts; ++iter) {
            chrono::steady_clock::time_point attempt_tm_start = chrono::steady_clock::now();
            mjData *d = mj_makeData(m);
            mj_step(m, d);

            double rotation = PI/8 * map_number;

            rotate(m, d, rotation);
            move_to(m, d, attempt_positions[iter]);
            move_down(m, d);
            grab(m, d);
            move_up(m, d);

            if (gripper_holds(m, d)) {
                success_map[attempt_positions[iter].first][attempt_positions[iter].second] = true;
                cout << "SUCCESS\n";
            }

            double attempt_time = get_tm_diff(attempt_tm_start);
            mj_deleteData(d);

            cout << "attempt " << iter + 1  << "/" << all_attempts << ", time = " << attempt_time << "\n\n";
        }

        double sim_time = get_tm_diff(sim_tm_start);
        cout << "Simulation time: " << sim_time << "\n";

        save_success_map(success_map, map_number);
    }
    
    mj_deleteModel(m);
    mj_deactivate();

    return 0;
}
