# cmake_minimum_required(VERSION 3.16)
# project(SP500 LANGUAGES CXX)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# add_executable(nn_cpp src/NeuralNetwork.cpp)
# add_executable(main_cpp src/main.cpp)
# target_link_libraries(nn_cpp PRIVATE Eigen3::Eigen)

cmake_minimum_required(VERSION 3.16)
project(SP500 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Add both main.cpp and NeuralNetwork.cpp
add_executable(nn_cpp
    src/main.cpp
    src/NeuralNetwork.cpp
)

target_link_libraries(nn_cpp PRIVATE Eigen3::Eigen)

# Optional: custom clean-all target for deep cleaning
add_custom_target(clean-all
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/*
  COMMENT "Removing all build artifacts"
)
