cmake_minimum_required(VERSION 2.8)
project(Detect)

set(OpenCV_STATIC ON)
set(Qt_DIR c:/Qt/4.8.0)

find_package( OpenCV REQUIRED)
set(BOOST_LIBRARYDIR C:/local/boost_1_55_0/lib64-msvc-11.0)
set(BOOST_INCLUDEDIR C:/local/boost_1_55_0)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC ON)
set(Boost_USE_MULTITHREAD )
set(Boost_NO_BOOST_CMAKE ON)

find_package(Boost REQUIRED filesystem) 
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})
add_executable(detect src/detect.cpp)
target_link_libraries( detect ${Boost_LIBRARIES} ${OpenCV_LIBS})