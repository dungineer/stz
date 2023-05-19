cmake_minimum_required(VERSION 3.20)
project(Detect)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(OpenCV_DIR "/home/nan17m521/build")
find_package(OpenCV REQUIRED PATHS /usr/local/share/opencv4 NO_DEFAULT_PATH)

add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

target_include_directories(${PROJECT_NAME} PRIVATE detect)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/sample_mpg.avi ${PROJECT_BINARY_DIR}/sample_mpg.avi)
