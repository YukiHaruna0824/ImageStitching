cmake_minimum_required(VERSION 2.8)

project(ImageStitch)

set(SRC_DIR ${PROJECT_SOURCE_DIR}/src/)
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/)
set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib/)

include_directories(${INCLUDE_DIR})
# include_directories(${LIB_DIR})

add_library(ImageUtils ${SRC_DIR}ImageUtils.h ${SRC_DIR}ImageUtils.cpp 
						${SRC_DIR}HarrisDetector.h ${SRC_DIR}HarrisDetector.cpp)
#add_library(MTBUtils ${SRC_DIR}MTBUtils.h ${SRC_DIR}MTBUtils.cpp)
add_executable(ImageStitch ${SRC_DIR}main.cpp)

target_link_libraries(ImageStitch ${LIB_DIR}opencv_world341.lib)
target_link_libraries(ImageStitch ImageUtils)
# target_link_libraries(ImageStitch ImageUtils MTBUtils)
