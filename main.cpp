#include<stdbool.h>
#include <math.h>

#include "mujoco.h"
#include "glfw3.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include <fstream>

//simulation end time
double simend = 50;

//state machine
int fsm_state;

#define start_state 0
#define move_x1 1
#define move_y1 2
#define move_z1 3
#define grip 4
#define move_z2 5
#define move_y2 6
#define move_x2 7
#define move_z3 8
#define let_go 9
#define move_z4 10

const double move_time = 2;

char path[] = "../myproject/mujoco-grasping-sim/";
char xmlfile[] = "gripper.xml";

const int WIDTH = 1244;
const int HEIGHT = 700;


// MuJoCo data structures
mjModel *m = NULL;                  // MuJoCo model
mjData *d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

// holders of one step history of time and position to calculate dertivatives
mjtNum position_history = 0;
mjtNum previous_time = 0;

// controller related variables
float_t ctrl_update_freq = 100;
mjtNum last_update = 0.0;
mjtNum ctrl;

// keyboard callback
void keyboard(GLFWwindow *window, int key, int scancode, int act, int mods) {
    // backspace: reset simulation
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}

// mouse button callback
void mouse_button(GLFWwindow *window, int button, int act, int mods) {
    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow *window, double xpos, double ypos) {
    // no buttons down: nothing to do
    if (!button_left && !button_middle && !button_right)
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (button_right)
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if (button_left)
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow *window, double xoffset, double yoffset) {
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

/******************************/
void set_torque_control(const mjModel *m, int actuator_no, int flag) {
    if (flag == 0)
        m->actuator_gainprm[10 * actuator_no + 0] = 0;
    else
        m->actuator_gainprm[10 * actuator_no + 0] = 1;
}
/******************************/


/******************************/
void set_position_servo(const mjModel *m, int actuator_no, double kp) {
    m->actuator_gainprm[10 * actuator_no + 0] = kp;
    m->actuator_biasprm[10 * actuator_no + 1] = -kp;
}
/******************************/

/******************************/
void set_velocity_servo(const mjModel *m, int actuator_no, double kv) {
    m->actuator_gainprm[10 * actuator_no + 0] = kv;
    m->actuator_biasprm[10 * actuator_no + 2] = -kv;
}
/******************************/

//**************************
void init_controller(const mjModel *m, mjData *d) {
    fsm_state = start_state;
}

//**************************
void mycontroller(const mjModel *m, mjData *d) {
    fsm_state = (int) floor(d->time / move_time);

    if (fsm_state == start_state) {

    }
    if (fsm_state == move_x1) {
        d->ctrl[0] = -0.5;
    }
    if (fsm_state == move_y1) {
        d->ctrl[1] = -1.2;
    }
    if (fsm_state == move_z1) {
        d->ctrl[2] = -2;
    }
    if (fsm_state == grip) {
        d->ctrl[3] = 0.5;
        d->ctrl[4] = 0.5;
    }
    if (fsm_state == move_z2) {
        d->ctrl[2] = 0;
    }
    if (fsm_state == move_y2) {
        d->ctrl[1] = 0.7;
    }
    if (fsm_state == move_x2) {
        d->ctrl[0] = 0;
    }
    if (fsm_state == move_z3) {
        d->ctrl[2] = -1.3;
    }
    if (fsm_state == let_go) {
        d->ctrl[3] = 0;
        d->ctrl[4] = 0;
    }
    if (fsm_state == move_z4) {
        d->ctrl[2] = 0;
    }
}

void make_tga_image(mjrRect viewport) {
    unsigned char *buffer = new unsigned char[WIDTH * HEIGHT * 3];
    mjr_readPixels(buffer, NULL, viewport, &con);

    FILE *out = fopen("../myproject/mujoco-grasping-sim/photo.tga", "w");
    short TGAhead[] = {0, 2, 0, 0, 0, 0, WIDTH, HEIGHT, 24};
    fwrite(&TGAhead, sizeof(TGAhead), 1, out);
    fwrite(buffer, 3 * WIDTH * HEIGHT, 1, out);
    fclose(out);

    delete[] buffer;
}


//************************
// main function
int main(int argc, const char **argv) {

    // activate software
    mj_activate("mjkey.txt");

    char xmlpath[100] = {};

    strcat(xmlpath, path);
    strcat(xmlpath, xmlfile);

    // load and compile model
    char error[1000] = "Could not load binary model";

    // check command-line arguments
    if (argc < 2)
        m = mj_loadXML(xmlpath, 0, error, 1000);

    else if (strlen(argv[1]) > 4 && !strcmp(argv[1] + strlen(argv[1]) - 4, ".mjb"))
        m = mj_loadModel(argv[1], 0);
    else
        m = mj_loadXML(argv[1], 0, error, 1000);
    if (!m)
        mju_error_s("Load model error: %s", error);

    // make data
    d = mj_makeData(m);


    // init GLFW
    if (!glfwInit())
        mju_error("Could not initialize GLFW");

    // create window, make OpenGL context current, request v-sync
    GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);                // space for 2000 objects
    mjr_makeContext(m, &con, mjFONTSCALE_150);   // model-specific context

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    double arr_view[] = {89.608063, -11.588379, 5, 0.000000, 0.000000, 1.000000};
    cam.azimuth = arr_view[0];
    cam.elevation = arr_view[1];
    cam.distance = arr_view[2];
    cam.lookat[0] = arr_view[3];
    cam.lookat[1] = arr_view[4];
    cam.lookat[2] = arr_view[5];

    // install control callback
    mjcb_control = mycontroller;
    init_controller(m, d);

    int photo_counter = 0;

    // use the first while condition if you want to simulate for a period.
    while (!glfwWindowShouldClose(window)) {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = d->time;
        while (d->time - simstart < 1.0 / 60.0) {
            mj_step(m, d);
        }

        if (d->time >= simend) {
            break;
        }

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);
        //printf("{%f, %f, %f, %f, %f, %f};\n",cam.azimuth,cam.elevation, cam.distance,cam.lookat[0],cam.lookat[1],cam.lookat[2]);

        if (photo_counter < 1) {
            make_tga_image(viewport);
            photo_counter++;
        }

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();

    }

    // free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();

    // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif

    return 1;
}
