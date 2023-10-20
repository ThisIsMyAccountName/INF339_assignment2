cmake_minimum_required(VERSION 3.15)

project(mesh_calculator
        VERSION 1.0
        DESCRIPTION "Mesh calculator"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3 -ffast-math")

find_package(Boost REQUIRED
             COMPONENTS
                 program_options
                 mpi
             REQUIRED)

add_executable(mesh_calc total.cpp)
target_include_directories(mesh_calc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(mesh_calc PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(mesh_calc PRIVATE ${Boost_LIBRARIES})
target_compile_options(mesh_calc PRIVATE #[[ -Wall -Wextra -Wpedantic -Werror ]])

set(CMAKE_COLOR_DIAGNOSTICS On)