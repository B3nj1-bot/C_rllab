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
include src/CMakeFiles/src.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/src.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/src.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/src.dir/flags.make

src/CMakeFiles/src.dir/ContinuousMLPQFunction.cpp.o: src/CMakeFiles/src.dir/flags.make
src/CMakeFiles/src.dir/ContinuousMLPQFunction.cpp.o: /Users/KevinChang/Documents/nsa/src/ContinuousMLPQFunction.cpp
src/CMakeFiles/src.dir/ContinuousMLPQFunction.cpp.o: src/CMakeFiles/src.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/KevinChang/Documents/nsa/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/src.dir/ContinuousMLPQFunction.cpp.o"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/src.dir/ContinuousMLPQFunction.cpp.o -MF CMakeFiles/src.dir/ContinuousMLPQFunction.cpp.o.d -o CMakeFiles/src.dir/ContinuousMLPQFunction.cpp.o -c /Users/KevinChang/Documents/nsa/src/ContinuousMLPQFunction.cpp

src/CMakeFiles/src.dir/ContinuousMLPQFunction.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/src.dir/ContinuousMLPQFunction.cpp.i"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/KevinChang/Documents/nsa/src/ContinuousMLPQFunction.cpp > CMakeFiles/src.dir/ContinuousMLPQFunction.cpp.i

src/CMakeFiles/src.dir/ContinuousMLPQFunction.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/src.dir/ContinuousMLPQFunction.cpp.s"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/KevinChang/Documents/nsa/src/ContinuousMLPQFunction.cpp -o CMakeFiles/src.dir/ContinuousMLPQFunction.cpp.s

src/CMakeFiles/src.dir/DeterministicMLPPolicy.cpp.o: src/CMakeFiles/src.dir/flags.make
src/CMakeFiles/src.dir/DeterministicMLPPolicy.cpp.o: /Users/KevinChang/Documents/nsa/src/DeterministicMLPPolicy.cpp
src/CMakeFiles/src.dir/DeterministicMLPPolicy.cpp.o: src/CMakeFiles/src.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/KevinChang/Documents/nsa/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/src.dir/DeterministicMLPPolicy.cpp.o"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/src.dir/DeterministicMLPPolicy.cpp.o -MF CMakeFiles/src.dir/DeterministicMLPPolicy.cpp.o.d -o CMakeFiles/src.dir/DeterministicMLPPolicy.cpp.o -c /Users/KevinChang/Documents/nsa/src/DeterministicMLPPolicy.cpp

src/CMakeFiles/src.dir/DeterministicMLPPolicy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/src.dir/DeterministicMLPPolicy.cpp.i"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/KevinChang/Documents/nsa/src/DeterministicMLPPolicy.cpp > CMakeFiles/src.dir/DeterministicMLPPolicy.cpp.i

src/CMakeFiles/src.dir/DeterministicMLPPolicy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/src.dir/DeterministicMLPPolicy.cpp.s"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/KevinChang/Documents/nsa/src/DeterministicMLPPolicy.cpp -o CMakeFiles/src.dir/DeterministicMLPPolicy.cpp.s

src/CMakeFiles/src.dir/LayerPowered.cpp.o: src/CMakeFiles/src.dir/flags.make
src/CMakeFiles/src.dir/LayerPowered.cpp.o: /Users/KevinChang/Documents/nsa/src/LayerPowered.cpp
src/CMakeFiles/src.dir/LayerPowered.cpp.o: src/CMakeFiles/src.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/KevinChang/Documents/nsa/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/src.dir/LayerPowered.cpp.o"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/src.dir/LayerPowered.cpp.o -MF CMakeFiles/src.dir/LayerPowered.cpp.o.d -o CMakeFiles/src.dir/LayerPowered.cpp.o -c /Users/KevinChang/Documents/nsa/src/LayerPowered.cpp

src/CMakeFiles/src.dir/LayerPowered.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/src.dir/LayerPowered.cpp.i"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/KevinChang/Documents/nsa/src/LayerPowered.cpp > CMakeFiles/src.dir/LayerPowered.cpp.i

src/CMakeFiles/src.dir/LayerPowered.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/src.dir/LayerPowered.cpp.s"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/KevinChang/Documents/nsa/src/LayerPowered.cpp -o CMakeFiles/src.dir/LayerPowered.cpp.s

src/CMakeFiles/src.dir/NormalizedEnv.cpp.o: src/CMakeFiles/src.dir/flags.make
src/CMakeFiles/src.dir/NormalizedEnv.cpp.o: /Users/KevinChang/Documents/nsa/src/NormalizedEnv.cpp
src/CMakeFiles/src.dir/NormalizedEnv.cpp.o: src/CMakeFiles/src.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/KevinChang/Documents/nsa/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/src.dir/NormalizedEnv.cpp.o"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/src.dir/NormalizedEnv.cpp.o -MF CMakeFiles/src.dir/NormalizedEnv.cpp.o.d -o CMakeFiles/src.dir/NormalizedEnv.cpp.o -c /Users/KevinChang/Documents/nsa/src/NormalizedEnv.cpp

src/CMakeFiles/src.dir/NormalizedEnv.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/src.dir/NormalizedEnv.cpp.i"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/KevinChang/Documents/nsa/src/NormalizedEnv.cpp > CMakeFiles/src.dir/NormalizedEnv.cpp.i

src/CMakeFiles/src.dir/NormalizedEnv.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/src.dir/NormalizedEnv.cpp.s"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/KevinChang/Documents/nsa/src/NormalizedEnv.cpp -o CMakeFiles/src.dir/NormalizedEnv.cpp.s

src/CMakeFiles/src.dir/Parameterized.cpp.o: src/CMakeFiles/src.dir/flags.make
src/CMakeFiles/src.dir/Parameterized.cpp.o: /Users/KevinChang/Documents/nsa/src/Parameterized.cpp
src/CMakeFiles/src.dir/Parameterized.cpp.o: src/CMakeFiles/src.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/KevinChang/Documents/nsa/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/src.dir/Parameterized.cpp.o"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/src.dir/Parameterized.cpp.o -MF CMakeFiles/src.dir/Parameterized.cpp.o.d -o CMakeFiles/src.dir/Parameterized.cpp.o -c /Users/KevinChang/Documents/nsa/src/Parameterized.cpp

src/CMakeFiles/src.dir/Parameterized.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/src.dir/Parameterized.cpp.i"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/KevinChang/Documents/nsa/src/Parameterized.cpp > CMakeFiles/src.dir/Parameterized.cpp.i

src/CMakeFiles/src.dir/Parameterized.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/src.dir/Parameterized.cpp.s"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/KevinChang/Documents/nsa/src/Parameterized.cpp -o CMakeFiles/src.dir/Parameterized.cpp.s

src/CMakeFiles/src.dir/SimpleReplayPool.cpp.o: src/CMakeFiles/src.dir/flags.make
src/CMakeFiles/src.dir/SimpleReplayPool.cpp.o: /Users/KevinChang/Documents/nsa/src/SimpleReplayPool.cpp
src/CMakeFiles/src.dir/SimpleReplayPool.cpp.o: src/CMakeFiles/src.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/KevinChang/Documents/nsa/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/CMakeFiles/src.dir/SimpleReplayPool.cpp.o"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/src.dir/SimpleReplayPool.cpp.o -MF CMakeFiles/src.dir/SimpleReplayPool.cpp.o.d -o CMakeFiles/src.dir/SimpleReplayPool.cpp.o -c /Users/KevinChang/Documents/nsa/src/SimpleReplayPool.cpp

src/CMakeFiles/src.dir/SimpleReplayPool.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/src.dir/SimpleReplayPool.cpp.i"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/KevinChang/Documents/nsa/src/SimpleReplayPool.cpp > CMakeFiles/src.dir/SimpleReplayPool.cpp.i

src/CMakeFiles/src.dir/SimpleReplayPool.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/src.dir/SimpleReplayPool.cpp.s"
	cd /Users/KevinChang/Documents/nsa/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/KevinChang/Documents/nsa/src/SimpleReplayPool.cpp -o CMakeFiles/src.dir/SimpleReplayPool.cpp.s

# Object files for target src
src_OBJECTS = \
"CMakeFiles/src.dir/ContinuousMLPQFunction.cpp.o" \
"CMakeFiles/src.dir/DeterministicMLPPolicy.cpp.o" \
"CMakeFiles/src.dir/LayerPowered.cpp.o" \
"CMakeFiles/src.dir/NormalizedEnv.cpp.o" \
"CMakeFiles/src.dir/Parameterized.cpp.o" \
"CMakeFiles/src.dir/SimpleReplayPool.cpp.o"

# External object files for target src
src_EXTERNAL_OBJECTS =

src/libsrc.a: src/CMakeFiles/src.dir/ContinuousMLPQFunction.cpp.o
src/libsrc.a: src/CMakeFiles/src.dir/DeterministicMLPPolicy.cpp.o
src/libsrc.a: src/CMakeFiles/src.dir/LayerPowered.cpp.o
src/libsrc.a: src/CMakeFiles/src.dir/NormalizedEnv.cpp.o
src/libsrc.a: src/CMakeFiles/src.dir/Parameterized.cpp.o
src/libsrc.a: src/CMakeFiles/src.dir/SimpleReplayPool.cpp.o
src/libsrc.a: src/CMakeFiles/src.dir/build.make
src/libsrc.a: src/CMakeFiles/src.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/KevinChang/Documents/nsa/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX static library libsrc.a"
	cd /Users/KevinChang/Documents/nsa/build/src && $(CMAKE_COMMAND) -P CMakeFiles/src.dir/cmake_clean_target.cmake
	cd /Users/KevinChang/Documents/nsa/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/src.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/src.dir/build: src/libsrc.a
.PHONY : src/CMakeFiles/src.dir/build

src/CMakeFiles/src.dir/clean:
	cd /Users/KevinChang/Documents/nsa/build/src && $(CMAKE_COMMAND) -P CMakeFiles/src.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/src.dir/clean

src/CMakeFiles/src.dir/depend:
	cd /Users/KevinChang/Documents/nsa/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/KevinChang/Documents/nsa /Users/KevinChang/Documents/nsa/src /Users/KevinChang/Documents/nsa/build /Users/KevinChang/Documents/nsa/build/src /Users/KevinChang/Documents/nsa/build/src/CMakeFiles/src.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/src.dir/depend

