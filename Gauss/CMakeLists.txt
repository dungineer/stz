cmake_minimum_required(VERSION 3.19)
project(List)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(googletest)

enable_testing()

add_executable(${PROJECT_NAME} tests.cpp)
target_link_libraries(${PROJECT_NAME} gtest gtest_main)

target_include_directories(${PROJECT_NAME} PUBLIC gauss)
target_include_directories(${PROJECT_NAME} PUBLIC googletest/googletest/include)
