cmake_minimum_required(VERSION 2.8)
project(FetureExt)

find_package( OpenCV REQUIRED)
set(BOOST_ROOT D:/workspace/3d/developer/lyingw/PointCloudStreamer/lib/boost_1_55_0_vs11_64)
set(Boost_USE_STATIC_LIBS ON) 
set(BOOST_LIBRARYDIR D:/workspace/3d/developer/lyingw/PointCloudStreamer/lib/boost_1_55_0_vs11_64/lib)

find_package(Boost REQUIRED filesystem) 
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})
add_executable(feature src/feature.cpp)
target_link_libraries( feature ${Boost_LIBRARIES} ${OpenCV_LIBS})