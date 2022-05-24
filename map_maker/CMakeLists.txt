cmake_minimum_required(VERSION 3.21)
project(mujoco_grasping_sim)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
        ${OPENGL_INCLUDE_DIR}
        ${GLEW_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        $ENV{MUJOCO_PATH}/include)

link_directories($ENV{MUJOCO_PATH}/bin)

file(GLOB LIB_MUJOCO $ENV{MUJOCO_PATH}/bin/libmujoco[0-9][0-9][0-9].so)

##Showing mujoco library found
message(STATUS "MuJoCo lib: " ${LIB_MUJOCO})

add_executable(mujoco_grasping_sim main.cpp)

target_link_libraries(mujoco_grasping_sim ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} ${LIB_MUJOCO})




