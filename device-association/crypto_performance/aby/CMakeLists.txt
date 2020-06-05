cmake_minimum_required(VERSION 3.10)

project(ABY LANGUAGES CXX)

add_executable(aby aby.cpp)
find_package(ABY QUIET)
target_link_libraries(aby ABY::aby)
