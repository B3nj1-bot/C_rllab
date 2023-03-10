# Path definitions.
set(lib_dir ${CMAKE_CURRENT_SOURCE_DIR})

# Add torch library.
set(torch_dir ${lib_dir}/libtorch)
file(GLOB_RECURSE torch_sources CONFIGURE_DEPENDS ${torch_dir}/*.h ${torch_dir}/*.cpp)
add_library(torch_dir ${torch_sources})
set_target_properties(torch PROPERTIES LINKER_LANGUAGE CXX)

# Add overall libs library.
add_library(libs INTERFACE)
target_include_directories(libs INTERFACE ${lib_dir})
target_link_libraries(libs INTERFACE libtorch)