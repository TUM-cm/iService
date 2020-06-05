cmake_minimum_required(VERSION 3.10)

project(SEAL LANGUAGES CXX)
find_package(Threads REQUIRED)

add_executable(seal seal.cpp)
find_package(SEAL 2.3.1 EXACT REQUIRED)
target_link_libraries(seal SEAL::seal)
