#include <algorithm>
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <string>
#include <chrono>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "mujoco.h"
#include "glfw3.h"

#include <FreeImage.h>

using namespace std;
using namespace cv;

#define debug false
#define PI 3.14159265358979323846

// model and per-thread data
mjModel* m = NULL;
mjData* d[64];

// per-thread statistics
int contacts[64];
int constraints[64];
double simtime[64];

const char project_path[] = "../myproject/mujoco-grasping-sim/";
const char xmlfile[] = "gripper.xml";

const int WIDTH = 1000;
const int HEIGHT = 1000;

// timer
chrono::system_clock::time_point tm_start;
mjtNum gettm(void)
{
    chrono::duration<double> elapsed = chrono::system_clock::now() - tm_start;
    return elapsed.count();
}


// deallocate and print message
int finish(const char* msg = NULL, mjModel* m = NULL)
{
    // deallocate model
    if( m )
        mj_deleteModel(m);
    mj_deactivate();

    // print message
    if( msg )
        printf("%s\n", msg);

    return 0;
}


// thread function
void thread_simulate(int id, int nstep)
{
    // clear statistics
    contacts[id] = 0;
    constraints[id] = 0;
    
    // run and time
    double start = gettm();
    for( int i=0; i<nstep; i++ )
    {
        // advance simulation
        mj_step(m, d[id]);

        // accumulate statistics
        contacts[id] += d[id]->ncon;
        constraints[id] += d[id]->nefc;
    }
    simtime[id] = gettm() - start;
}




//////////////////// Color and Coutour Detection  //////////////////////
// Takes path to existing .png file and returns all relative positions 
// of element's centers. 
// 
// Input  : path to file.
// Output : list of (x, y) pairs.
vector<pair<int, int>> getContours(const string &png_path) {
    Mat img;
    try {
        img = imread(png_path);
    } catch (const Exception &e) {
        vector<pair<int, int>> empty{};
        return empty;
    }
    Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;

    // Preprocessing
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
    Canny(imgBlur, imgCanny, 25, 75);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imgCanny, imgDil, kernel);

    // Main function that finds contours.
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //drawContours(img, contours, -1, Scalar(255, 0, 255), 2);

    vector<pair<int, int> > result;

    vector<vector<Point> > conPoly(contours.size());
    vector<Rect> boundRect(contours.size());

    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        string objectType;

        // cout << area << endl;
        if (area > 1000) {
            double peri = arcLength(contours[i], true);
            approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
            // cout << conPoly[i].size() << endl;
            boundRect[i] = boundingRect(conPoly[i]);

            int objCor = (int) conPoly[i].size();

            // Tries to guess which polygon it is.
            if (objCor == 3) 
                objectType = "Triangle";
            else if (objCor == 4) {
                float aspRatio = (float) boundRect[i].width / (float) boundRect[i].height;
                if (aspRatio > 0.95 && aspRatio < 1.05)
                    objectType = "Square";
                else
                    objectType = "Rect";
            } else if (objCor < 7)
                objectType = std::to_string(objCor) + " corners";
            else
                objectType = "Circle?";

            // Draws back found contours on the image.
            drawContours(img, conPoly, i, Scalar(255, 0, 255), 2);
            // Draws area on which polygon was found.
            rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5);

            Point center = {(boundRect[i].tl().x + boundRect[i].br().x) / 2,
                            (boundRect[i].tl().y + boundRect[i].br().y) / 2};
            result.emplace_back(center.x, center.y);

            // Puts dot on center and objectType on top of polygon.
            putText(img, ".", center, FONT_HERSHEY_PLAIN, 1, Scalar(0, 100, 255), 2);
            putText(img, objectType, {boundRect[i].x, boundRect[i].y - 5}, FONT_HERSHEY_PLAIN, 1, Scalar(0, 69, 255),
                    2);
        }
    }

    if (debug) {
        cout << "center points" << endl;
        for (pair<int, int> &pair: result) {
            cout << pair.first << " " << pair.second << endl;
        }
        namedWindow("Image with contours", WINDOW_AUTOSIZE);
        imshow("Image with contours", img);
        waitKey();
    }

    return result;
}


// Searches element id through all elements in the model.
// Input : all models object, type of element, name of element in the xml
// Output : Element's id.
// Throws error when element is not found within the xml.
int get_id(const mjModel *m, int object, const char* name) {
    printf("-> -%s-\n", name);
    int object_id = mj_name2id(m, object, name);
    printf("aaa\n");
    if (object_id == -1) {
        printf("xle\n");
        throw std::runtime_error("mj_name2id error");
    }
    printf("bbb\n");
    return object_id;
}


void make_png_image(const mjModel *m, mjData *d, const mjvOption *opt, mjvScene *scn, const mjrContext *con,
                            GLFWwindow *window, mjvCamera *gripper_cam, const mjrRect viewport) {
    mj_step(m, d);
    int buffer_size = WIDTH * HEIGHT * 3;
    auto *buffer = new unsigned char[buffer_size];

    // Set alpha of both arms to 0.
    auto arm1 = get_id(m, mjOBJ_GEOM, "arm_left");
    auto arm2 = get_id(m, mjOBJ_GEOM, "arm_right");
    m->geom_rgba[4*arm1 + 3] = 0;
    m->geom_rgba[4*arm2 + 3] = 0;
    mjv_updateScene(m, d, opt, nullptr, gripper_cam, mjCAT_ALL, scn);
    mjr_render(viewport, scn, con);

    // Takes a picture.
    mjr_readPixels(buffer, nullptr, viewport, con);

    // Convert from BGR to RGB.
    for (int x = 0; x < HEIGHT; x++) {
        for (int y = 0; y < WIDTH; y++) {
            std::swap(buffer[x * WIDTH * 3 + y * 3], buffer[x * WIDTH * 3 + 2 + y * 3]);
        }
    }
    FIBITMAP *image = FreeImage_ConvertFromRawBits(buffer, WIDTH, HEIGHT, 3 * WIDTH, 24, 0x0000FF, 0xFF0000, 0x00FF00,
                                                   false);
    FreeImage_Save(FIF_PNG, image, "../myproject/mujoco-grasping-sim/photo.png", 0);

    // Set alpha of both arms back to 1.
    m->geom_rgba[4*arm1 + 3] = 1;
    m->geom_rgba[4*arm2 + 3] = 1;
    mjv_updateScene(m, d, opt, nullptr, gripper_cam, mjCAT_ALL, scn);
    mjr_render(viewport, scn, con);
    delete[] buffer;
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
    return abs(abs(d->xpos[3 * left_gripper_id]) - abs(d->xpos[3 * right_gripper_id])) > 0.02;
}

pair<mjtNum, mjtNum> get_body_coords(const mjModel *m, pair<int, int> pixel_coords,
                                     tuple<mjtNum, mjtNum, mjtNum> cam_coords) {
    if (HEIGHT != WIDTH) {
        throw std::runtime_error("get_body_coords error: WIDTH and HEIGHT has to be the same");
    }

    int cam_id = get_id(m, mjOBJ_CAMERA, "gripper-cam");

    mjtNum y_coord_diff = tan((m->cam_fovy[cam_id] / 2) * (PI / 180)) * get<2>(cam_coords) * 2;
    mjtNum x_coord_diff = y_coord_diff;

    mjtNum x_coord = get<0>(cam_coords) + (static_cast<double>(pixel_coords.first) - static_cast<double>(WIDTH) / 2)
                                          / WIDTH * x_coord_diff;
    mjtNum y_coord = get<1>(cam_coords) - (static_cast<double>(pixel_coords.second) - static_cast<double>(HEIGHT) / 2)
                                          / HEIGHT * y_coord_diff;

    //cout << "coords " << x_coord << " " << y_coord << endl;
    return make_pair(x_coord, y_coord);
}


pair<int, int> get_closest_pixels(const vector<pair<int, int>> &vec) {
    if (vec.empty()) {
        throw std::runtime_error("get_closest_pixels error: vec can't be empty");
    }

    double current_distance;
    double min_distance = DBL_MAX;
    pair<int, int> min_pair = make_pair(0, 0);

    for (pair<int, int> pair: vec) {
        current_distance = hypot((pair.first - WIDTH/2), (pair.second - HEIGHT/2));

        if (current_distance < min_distance) {
            min_distance = current_distance;
            min_pair = pair;
        }
    }

    return min_pair;
}


void simulate(mjtNum sim_time, const mjModel *m, mjData *d, const mjvOption *opt, mjvScene *scn, const mjrContext *con,
              GLFWwindow *window, mjvCamera *gripper_cam, const mjrRect viewport) {

    mjtNum sim_start = d->time;
    while (d->time < sim_start + sim_time) {
        mjtNum frame_sim_start = d->time;
        while (d->time - frame_sim_start < 1.0 / 60.0) {
            mj_step(m, d);
        }

        mjv_updateScene(m, d, opt, nullptr, gripper_cam, mjCAT_ALL, scn);
        mjr_render(viewport, scn, con);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}


void move_vertical_to_block(const mjModel *m, mjData *d, const mjvOption *opt, mjvScene *scn, const mjrContext *con,
                            GLFWwindow *window, mjvCamera *gripper_cam, const mjrRect viewport, vector<pair<int, int>> centers) {

    pair<int, int> closest_pixels = get_closest_pixels(centers);

    tuple<mjtNum, mjtNum, mjtNum> cam_coords = get_gripper_cam_coords(m, d);
    pair<mjtNum, mjtNum> closest_coords = get_body_coords(m, closest_pixels, cam_coords);

    d->ctrl[0] = closest_coords.first;
    d->ctrl[1] = closest_coords.second;

    simulate(2, m, d, opt, scn, con, window, gripper_cam, viewport);
}


void move_vertical_to_center(const mjModel *m, mjData *d, const mjvOption *opt, mjvScene *scn, const mjrContext *con,
                             GLFWwindow *window, mjvCamera *gripper_cam, const mjrRect viewport) {

    d->ctrl[0] = 0;
    d->ctrl[1] = 0;
    simulate(4, m, d, opt, scn, con, window, gripper_cam, viewport);
}


void move_vertical_to_container(const mjModel *m, mjData *d, const mjvOption *opt, mjvScene *scn, const mjrContext *con,
                                GLFWwindow *window, mjvCamera *gripper_cam, const mjrRect viewport) {

    d->ctrl[0] = 5;
    d->ctrl[1] = 5;
    simulate(4, m, d, opt, scn, con, window, gripper_cam, viewport);
}


void move_down(const mjModel *m, mjData *d, const mjvOption *opt, mjvScene *scn, const mjrContext *con,
               GLFWwindow *window, mjvCamera *gripper_cam, const mjrRect viewport) {
    d->ctrl[2] = -3.5;
    simulate(2, m, d, opt, scn, con, window, gripper_cam, viewport);
}


void move_up(const mjModel *m, mjData *d, const mjvOption *opt, mjvScene *scn, const mjrContext *con,
             GLFWwindow *window, mjvCamera *gripper_cam, const mjrRect viewport) {

    d->ctrl[2] = 0;
    simulate(2, m, d, opt, scn, con, window, gripper_cam, viewport);
}


void grab(const mjModel *m, mjData *d, const mjvOption *opt, mjvScene *scn, const mjrContext *con,
          GLFWwindow *window, mjvCamera *gripper_cam, const mjrRect viewport) {

    d->ctrl[3] = 0.5;
    d->ctrl[4] = 0.5;
    simulate(2, m, d, opt, scn, con, window, gripper_cam, viewport);
}


void release(const mjModel *m, mjData *d, const mjvOption *opt, mjvScene *scn, const mjrContext *con,
             GLFWwindow *window, mjvCamera *gripper_cam, const mjrRect viewport) {

    d->ctrl[3] = 0;
    d->ctrl[4] = 0;
    simulate(2, m, d, opt, scn, con, window, gripper_cam, viewport);
}

// main function
int main(int argc, const char** argv)
{
    char xmlpath[100] = {};
    strcat(xmlpath, project_path);
    strcat(xmlpath, xmlfile);

    char error[1000] = "Could not load binary model";
    printf("plik:%s.\n", xmlpath);
    struct stat buffer;
    if(stat (xmlpath, &buffer) != 0){
        char* cwd = getcwd(NULL, 0);
        printf("sciezka-%s-\n", cwd);
        printf("o nie\n");
        return 0;
    }
    m = mj_loadXML(xmlpath, 0, error, 1000);
    if(m == NULL){
        printf("bleee\n");
        return 1;
    }
    const char* name = "gripper-cam";
    printf("%d\n", mj_name2id(m, mjOBJ_CAMERA, name));

    if (!glfwInit())
        mju_error("Could not initialize GLFW");

    // create window, make OpenGL context current, request v-sync
    GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "Demo", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjvOption opt;
    mjvScene scn;
    mjrContext con;
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);                // space for 2000 objects
    mjr_makeContext(m, &con, mjFONTSCALE_150);   // model-specific context

    mjvCamera gripper_cam;
    gripper_cam.type = mjCAMERA_FIXED;
    printf("tutaj\n");
    printf("-%d-\n", mjOBJ_CAMERA);
    gripper_cam.fixedcamid = mj_name2id(m, mjOBJ_CAMERA, "gripper-cam");
    printf("aaa\n");
    if (gripper_cam.fixedcamid == -1) {
        printf("xle\n");
        throw std::runtime_error("mj_name2id error");
    }
    printf("tu\n");
    gripper_cam.trackbodyid = -1;

    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // activate MuJoCo Pro license (this must be *your* activation key)
    mj_activate("mjkey.txt");

    // read nstep and nthread
    int nstep = 0, nthread = 0, profile = 0;
    if( sscanf(argv[1], "%d", &nstep)!=1 || nstep<=0 )
        return finish("Invalid nstep argument");
    if( argc>3 )
        if( sscanf(argv[2], "%d", &nthread)!=1 )
            return finish("Invalid nthread argument");
    if( argc>4 )
        if( sscanf(argv[3], "%d", &profile)!=1 )
            return finish("Invalid profile argument");

    // clamp nthread to [1, 64]
    nthread = mjMAX(1, mjMIN(64, nthread));

    // get filename, determine file type
    std::string filename(argv[1]);
    bool binary = (filename.find(".mjb")!=std::string::npos);

    // make per-thread data
    int testkey = mj_name2id(m, mjOBJ_KEY, "test");
    for( int id=0; id<nthread; id++ )
    {
        d[id] = mj_makeData(m);
        if( !d[id] )
            return finish("Could not allocate mjData", m);

        // init to keyframe "test" if present
        if( testkey>=0 )
        {
            mju_copy(d[id]->qpos, m->key_qpos + testkey*m->nq, m->nq);
            mju_copy(d[id]->qvel, m->key_qvel + testkey*m->nv, m->nv);
            mju_copy(d[id]->act,  m->key_act  + testkey*m->na, m->na);
        }
    }

    // install timer callback for profiling if requested
    tm_start = chrono::system_clock::now();
    if( profile )
        mjcb_time = gettm;

    // print start
    if( nthread>1 )
        printf("\nRunning %d steps per thread at dt = %g ...\n\n", nstep, m->opt.timestep);
    else
        printf("\nRunning %d steps at dt = %g ...\n\n", nstep, m->opt.timestep);

    // run simulation, record total time
    thread th[64];
    double starttime = gettm();
    for( int id=0; id<nthread; id++ )
        th[id] = thread(thread_simulate, id, nstep);
    /*while (true) {
        mj_step(m, d);
        make_png_image(m, d, &opt, &scn, &con, window, &gripper_cam, viewport);
        auto centers = getContours("../myproject/mujoco-grasping-sim/photo.png");
        if (centers.size() == 0) {
            cout << "Moved all elements, shutting down\n";
            namedWindow("Moved all (seen) elements, shutting down\n"); // or some other indicator that robot finished.
            waitKey();
            break;
        }

        move_vertical_to_block(m, d, &opt, &scn, &con, window, &gripper_cam, viewport, centers);
        move_down(m, d, &opt, &scn, &con, window, &gripper_cam, viewport);
        grab(m, d, &opt, &scn, &con, window, &gripper_cam, viewport);
        move_up(m, d, &opt, &scn, &con, window, &gripper_cam, viewport);

        // Gripper may sometimes fail to grab elements.
        if (gripper_holds(m, d)) {
            move_vertical_to_container(m, d, &opt, &scn, &con, window, &gripper_cam, viewport);
        }

        release(m, d, &opt, &scn, &con, window, &gripper_cam, viewport);
        move_vertical_to_center(m, d, &opt, &scn, &con, window, &gripper_cam, viewport);
    }*/
    for( int id=0; id<nthread; id++ )
        th[id].join();
    double tottime = gettm() - starttime;

    // all-thread summary
    if( nthread>1 )
    {
        printf("Summary for all %d threads\n\n", nthread);
        printf(" Total simulation time  : %.2f s\n", tottime);
        printf(" Total steps per second : %.0f\n", nthread*nstep/tottime);
        printf(" Total realtime factor  : %.2f x\n", nthread*nstep*m->opt.timestep/tottime);
        printf(" Total time per step    : %.4f ms\n\n", 1000*tottime/(nthread*nstep));

        printf("Details for thread 0\n\n");
    }

    // details for thread 0
    printf(" Simulation time      : %.2f s\n", simtime[0]);
    printf(" Steps per second     : %.0f\n", nstep/simtime[0]);
    printf(" Realtime factor      : %.2f x\n", nstep*m->opt.timestep/simtime[0]);
    printf(" Time per step        : %.4f ms\n\n", 1000*simtime[0]/nstep);
    printf(" Contacts per step    : %d\n", contacts[0]/nstep);
    printf(" Constraints per step : %d\n", constraints[0]/nstep);
    printf(" Degrees of freedom   : %d\n\n", m->nv);

    // profiler results for thread 0
    if( profile )
    {
        printf(" Profiler phase (ms per step)\n");
        mjtNum tstep = d[0]->timer[mjTIMER_STEP].duration/d[0]->timer[mjTIMER_STEP].number;
        for( int i=0; i<mjNTIMER; i++ )
            if( d[0]->timer[i].number>0 )
            {
                mjtNum istep = d[0]->timer[i].duration/d[0]->timer[i].number;
                printf(" %16s : %.5f  (%6.2f %%)\n", mjTIMERSTRING[i],
                    1000*istep, 100*istep/tstep);
            }
    }

    // free per-thread data
    for( int id=0; id<nthread; id++ )
        mj_deleteData(d[id]);

    // free visualization storage
    glfwDestroyWindow(window);
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteModel(m);
    mj_deactivate();

    // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif

    // finalize
    return finish();
}