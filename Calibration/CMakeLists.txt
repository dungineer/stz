cmake_minimum_required(VERSION 3.19)
project(Calibration)

set(CMAKE_CXX_STANDARD 17)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries (${PROJECT_NAME} Eigen3::Eigen)

add_custom_command(TARGET ${PROJECT_NAME}
	POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/data/data.txt ${PROJECT_BINARY_DIR}/data.txt)

add_custom_target(latex
	COMMAND pdflatex README.tex
	COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/latex/README.pdf ${PROJECT_SOURCE_DIR}/README.pdf
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/latex)
