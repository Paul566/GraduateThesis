cmake_minimum_required(VERSION 3.25)
project(GeneralSolver)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH "/home/paul/Documents/GraduateThesis/GeneralSolver/or-tools_x86_64_Ubuntu-20.04_cpp_v9.6.2534")
find_package(re2)
find_package(ortools REQUIRED CONFIG)
include_directories(${ortools_INCLUDE_DIRS})

add_executable(GeneralSolver main.cpp TestReader.h GeneralSolver.h TestReader.cpp GeneralSolver.cpp
        SimplexInBallTestReader.h SimplexInBallTestReader.cpp)

target_link_libraries(GeneralSolver ortools::ortools)