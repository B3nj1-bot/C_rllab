cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(nsa)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
link_directories(/Users/KevinChang/opt/anaconda3/pkgs/mkl-2021.4.0-hecd8cb5_637/lib)
set(CMAKE_PREFIX_PATH /Users/KevinChang/opt/anaconda3/lib/python3.9/site-packages/torch/share/cmake
)

find_package(Torch REQUIRED)

set(src_dir ${CMAKE_SOURCE_DIR}/src)
set(entry_dir ${CMAKE_SOURCE_DIR}/entry)
add_subdirectory(${src_dir})

add_executable(ddpg entry/ddpg.cpp)
target_link_libraries(ddpg PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(ddpg PRIVATE src)
set_property(TARGET ddpg PROPERTY CXX_STANDARD 14)
