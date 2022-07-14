#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

#include "mujoco.h"

using namespace cv;
using namespace std;

#define PI 3.14159265358979323846

const char project_path[] = "";
const char xmlfile[] = "gripper.xml";
const size_t error_sz = 1024;
const size_t filename_sz = 100;
const int white_color = 255;
const int gripper_rotation = 5;

#define WIDTH 10
#define HEIGHT 10

#define ROTATIONS 8

#define debug false

mjtNum get_tm_diff(chrono::steady_clock::time_point tm_start) {
  chrono::duration<double> elapsed = chrono::steady_clock::now() - tm_start;
  return elapsed.count();
}

// Searches element id through all elements in the model.
// Input : all models object, type of element, name of element in the xml
// Output : Element's id.
// Throws error when element is not found within the xml.
int get_id(const mjModel *model, int object, const char *name) {
  int object_id = mj_name2id(model, object, name);
  if (object_id == -1) {
    throw std::runtime_error("mj_name2id error");
  }
  return object_id;
}

tuple<mjtNum, mjtNum, mjtNum> get_gripper_cam_coords(const mjModel *model,
                                                     mjData *data) {
  auto cam_body_id = get_id(model, mjOBJ_BODY, "gripper-cam-body");
  return make_tuple(data->xpos[3 * cam_body_id],
                    data->xpos[3 * cam_body_id + 1],
                    data->xpos[3 * cam_body_id + 2]);
}

// Checks whether gripper holds an object.
// Input : All models object, data of where models are located
// Returns true if gripper holds an object, false otherwise.
bool gripper_holds(const mjModel *model, mjData *data) {
  const double delta = 0.1;
  auto left_gripper_id = get_id(model, mjOBJ_BODY, "arm1_body");
  auto right_gripper_id = get_id(model, mjOBJ_BODY, "arm2_body");
  // Could not find a better solution as of now.
  auto diff = hypot(
      (data->xpos[3 * left_gripper_id] - data->xpos[3 * right_gripper_id]),
      (data->xpos[3 * left_gripper_id + 1] -
       data->xpos[3 * right_gripper_id + 1]));

  if (debug) {
    cout << data->xpos[3 * left_gripper_id] << " "
         << data->xpos[3 * left_gripper_id + 1] << " "
         << data->xpos[3 * left_gripper_id + 2] << " | "
         << data->xpos[3 * right_gripper_id] << " "
         << data->xpos[3 * right_gripper_id + 1] << " "
         << data->xpos[3 * right_gripper_id + 2] << "\n";
    cout << "diff = " << diff << "\n";
  }

  return diff > delta;
}

pair<mjtNum, mjtNum> get_body_coords(const mjModel *model,
                                     pair<int, int> pixel_height_width,
                                     tuple<mjtNum, mjtNum, mjtNum> cam_coords) {
  if (HEIGHT != WIDTH) {
    throw std::runtime_error(
        "get_body_coords error: WIDTH and HEIGHT has to be the same");
  }

  const int deg_to_rad = PI / 180;
  int cam_id = get_id(model, mjOBJ_CAMERA, "gripper-cam");

  mjtNum y_coord_diff =
      tan((model->cam_fovy[cam_id] / 2) * deg_to_rad) * get<2>(cam_coords) * 2;
  mjtNum x_coord_diff = y_coord_diff;

  mjtNum x_coord =
      get<0>(cam_coords) + (static_cast<double>(pixel_height_width.second) -
                            static_cast<double>(WIDTH) / 2) /
                               WIDTH * x_coord_diff;
  mjtNum y_coord =
      get<1>(cam_coords) - (static_cast<double>(pixel_height_width.first) -
                            static_cast<double>(HEIGHT) / 2) /
                               HEIGHT * y_coord_diff;

  return make_pair(x_coord, y_coord);
}

void simulate(mjtNum sim_time, const mjModel *model, mjData *data) {
  mjtNum sim_start = data->time;
  while (data->time < sim_start + sim_time) {
    // it used to be "frame based"
    // mjtNum frame_sim_start = data->time;
    // while (data->time - frame_sim_start < 1.0 / 60.0) {
    mj_step(model, data);
    //}
  }
}

void move_to(const mjModel *model, mjData *data, pair<int, int> goal) {
  tuple<mjtNum, mjtNum, mjtNum> cam_coords =
      get_gripper_cam_coords(model, data);
  pair<mjtNum, mjtNum> closest_coords =
      get_body_coords(model, goal, cam_coords);

  data->ctrl[0] = closest_coords.first;
  data->ctrl[1] = closest_coords.second;

  simulate(1, model, data);
}

void move_down(const mjModel *model, mjData *data) {
  const double downZ = -3.5;
  data->ctrl[2] = downZ;
  simulate(2, model, data);
}

void move_up(const mjModel *model, mjData *data) {
  data->ctrl[2] = 0;
  simulate(2, model, data);
}

void grab(const mjModel *model, mjData *data) {
  const double closeClamps = 0.5;
  data->ctrl[3] = closeClamps;
  data->ctrl[4] = closeClamps;
  simulate(2, model, data);
}

void rotate(const mjModel *model, mjData *data, double radian_angle) {
  data->ctrl[gripper_rotation] = radian_angle;
}

vector<pair<int, int>> get_attempt_positions() {
  vector<pair<int, int>> positions;

  for (int i = 0; i < HEIGHT; ++i) {
    for (int j = 0; j < WIDTH; ++j) {
      positions.emplace_back(i, j);
    }
  }
  return positions;
}

void save_success_map(const bool arr[HEIGHT][WIDTH], int map_number) {
  // Crete the image base
  Mat image = Mat::zeros(HEIGHT, WIDTH, CV_8U);

  // Check for failure
  if (image.empty()) {
    printf("Could not create the image!!\n");
    return;
  }

  // Put correct values of pixels
  for (int i = 0; i < HEIGHT; ++i) {
    for (int j = 0; j < WIDTH; ++j) {
      image.at<unsigned char>(i, j) = white_color * static_cast<int>(arr[i][j]);
    }
  }

  string path =
      string(project_path) + "success_map" + to_string(map_number) + ".png";
  imwrite(path, image);
}

void make_success_map(const mjModel *model, int map_number,
                      const vector<pair<int, int>> &attempt_positions) {
  chrono::steady_clock::time_point sim_tm_start = chrono::steady_clock::now();

  bool success_map[HEIGHT][WIDTH];

  for (int i = 0; i < HEIGHT; ++i) {
    for (int j = 0; j < WIDTH; ++j) {
      success_map[i][j] = false;
    }
  }

  unsigned long all_attempts = attempt_positions.size();

  for (int iter = 0; iter < all_attempts; ++iter) {
    chrono::steady_clock::time_point attempt_tm_start =
        chrono::steady_clock::now();
    mjData *data = mj_makeData(model);
    mj_step(model, data);

    double rotation = PI / ROTATIONS * map_number;

    rotate(model, data, rotation);
    move_to(model, data, attempt_positions[iter]);
    move_down(model, data);
    grab(model, data);
    move_up(model, data);

    if (gripper_holds(model, data)) {
      success_map[attempt_positions[iter].first]
                 [attempt_positions[iter].second] = true;
      cout << "SUCCESS\n";
    }

    double attempt_time = get_tm_diff(attempt_tm_start);
    mj_deleteData(data);

    cout << "map " << map_number + 1 << "/" << ROTATIONS << endl;
    cout << "attempt " << iter + 1 << "/" << all_attempts
         << ", time = " << attempt_time << "\n\n";
  }

  double sim_time = get_tm_diff(sim_tm_start);
  cout << "Simulation time: " << sim_time << "\n";

  save_success_map(success_map, map_number);
}

int main() {
  char xmlpath[filename_sz] = {};
  strcat(xmlpath, project_path);
  strcat(xmlpath, xmlfile);

  char error[error_sz];
  mjModel *model = mj_loadXML(xmlpath, nullptr, error, error_sz);
  if (model == nullptr) {
    std::cout << error;
    return -1;
  }
  mjvCamera gripper_cam;
  gripper_cam.type = mjCAMERA_FIXED;
  gripper_cam.fixedcamid = get_id(model, mjOBJ_CAMERA, "gripper-cam");
  gripper_cam.trackbodyid = -1;

  vector<pair<int, int>> attempt_positions = get_attempt_positions();

  for (int map_number = 0; map_number < ROTATIONS; map_number++) {
    make_success_map(model, map_number, attempt_positions);
  }

  mj_deleteModel(model);
  mj_deactivate();

  return 0;
}
