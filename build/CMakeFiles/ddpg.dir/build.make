# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.25.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.25.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/KevinChang/Documents/nsa

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/KevinChang/Documents/nsa/build

# Include any dependencies generated for this target.
include CMakeFiles/ddpg.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ddpg.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ddpg.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ddpg.dir/flags.make

CMakeFiles/ddpg.dir/entry/ddpg.cpp.o: CMakeFiles/ddpg.dir/flags.make
CMakeFiles/ddpg.dir/entry/ddpg.cpp.o: /Users/KevinChang/Documents/nsa/entry/ddpg.cpp
CMakeFiles/ddpg.dir/entry/ddpg.cpp.o: CMakeFiles/ddpg.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/KevinChang/Documents/nsa/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ddpg.dir/entry/ddpg.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ddpg.dir/entry/ddpg.cpp.o -MF CMakeFiles/ddpg.dir/entry/ddpg.cpp.o.d -o CMakeFiles/ddpg.dir/entry/ddpg.cpp.o -c /Users/KevinChang/Documents/nsa/entry/ddpg.cpp

CMakeFiles/ddpg.dir/entry/ddpg.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ddpg.dir/entry/ddpg.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/KevinChang/Documents/nsa/entry/ddpg.cpp > CMakeFiles/ddpg.dir/entry/ddpg.cpp.i

CMakeFiles/ddpg.dir/entry/ddpg.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ddpg.dir/entry/ddpg.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/KevinChang/Documents/nsa/entry/ddpg.cpp -o CMakeFiles/ddpg.dir/entry/ddpg.cpp.s

# Object files for target ddpg
ddpg_OBJECTS = \
"CMakeFiles/ddpg.dir/entry/ddpg.cpp.o"

# External object files for target ddpg
ddpg_EXTERNAL_OBJECTS =

ddpg: CMakeFiles/ddpg.dir/entry/ddpg.cpp.o
ddpg: CMakeFiles/ddpg.dir/build.make
ddpg: /Users/KevinChang/opt/anaconda3/lib/python3.9/site-packages/torch/lib/libc10.dylib
ddpg: src/libsrc.a
ddpg: /Users/KevinChang/opt/anaconda3/lib/python3.9/site-packages/torch/lib/libtorch.dylib
ddpg: /Users/KevinChang/opt/anaconda3/lib/python3.9/site-packages/torch/lib/libtorch_cpu.dylib
ddpg: /Users/KevinChang/opt/anaconda3/lib/python3.9/site-packages/torch/lib/libc10.dylib
ddpg: /Users/KevinChang/opt/anaconda3/lib/python3.9/site-packages/torch/lib/libc10.dylib
ddpg: CMakeFiles/ddpg.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/KevinChang/Documents/nsa/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ddpg"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ddpg.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ddpg.dir/build: ddpg
.PHONY : CMakeFiles/ddpg.dir/build

CMakeFiles/ddpg.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ddpg.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ddpg.dir/clean

CMakeFiles/ddpg.dir/depend:
	cd /Users/KevinChang/Documents/nsa/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/KevinChang/Documents/nsa /Users/KevinChang/Documents/nsa /Users/KevinChang/Documents/nsa/build /Users/KevinChang/Documents/nsa/build /Users/KevinChang/Documents/nsa/build/CMakeFiles/ddpg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ddpg.dir/depend
