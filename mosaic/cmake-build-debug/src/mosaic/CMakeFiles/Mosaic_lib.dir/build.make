# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

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

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/normadi/Downloads/mosaic

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/normadi/Downloads/mosaic/cmake-build-debug

# Include any dependencies generated for this target.
include src/mosaic/CMakeFiles/Mosaic_lib.dir/depend.make

# Include the progress variables for this target.
include src/mosaic/CMakeFiles/Mosaic_lib.dir/progress.make

# Include the compile flags for this target's objects.
include src/mosaic/CMakeFiles/Mosaic_lib.dir/flags.make

src/mosaic/CMakeFiles/Mosaic_lib.dir/mosaic.cpp.o: src/mosaic/CMakeFiles/Mosaic_lib.dir/flags.make
src/mosaic/CMakeFiles/Mosaic_lib.dir/mosaic.cpp.o: ../src/mosaic/mosaic.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/normadi/Downloads/mosaic/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/mosaic/CMakeFiles/Mosaic_lib.dir/mosaic.cpp.o"
	cd /Users/normadi/Downloads/mosaic/cmake-build-debug/src/mosaic && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Mosaic_lib.dir/mosaic.cpp.o -c /Users/normadi/Downloads/mosaic/src/mosaic/mosaic.cpp

src/mosaic/CMakeFiles/Mosaic_lib.dir/mosaic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Mosaic_lib.dir/mosaic.cpp.i"
	cd /Users/normadi/Downloads/mosaic/cmake-build-debug/src/mosaic && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/normadi/Downloads/mosaic/src/mosaic/mosaic.cpp > CMakeFiles/Mosaic_lib.dir/mosaic.cpp.i

src/mosaic/CMakeFiles/Mosaic_lib.dir/mosaic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Mosaic_lib.dir/mosaic.cpp.s"
	cd /Users/normadi/Downloads/mosaic/cmake-build-debug/src/mosaic && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/normadi/Downloads/mosaic/src/mosaic/mosaic.cpp -o CMakeFiles/Mosaic_lib.dir/mosaic.cpp.s

# Object files for target Mosaic_lib
Mosaic_lib_OBJECTS = \
"CMakeFiles/Mosaic_lib.dir/mosaic.cpp.o"

# External object files for target Mosaic_lib
Mosaic_lib_EXTERNAL_OBJECTS =

src/mosaic/libMosaic_lib.a: src/mosaic/CMakeFiles/Mosaic_lib.dir/mosaic.cpp.o
src/mosaic/libMosaic_lib.a: src/mosaic/CMakeFiles/Mosaic_lib.dir/build.make
src/mosaic/libMosaic_lib.a: src/mosaic/CMakeFiles/Mosaic_lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/normadi/Downloads/mosaic/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libMosaic_lib.a"
	cd /Users/normadi/Downloads/mosaic/cmake-build-debug/src/mosaic && $(CMAKE_COMMAND) -P CMakeFiles/Mosaic_lib.dir/cmake_clean_target.cmake
	cd /Users/normadi/Downloads/mosaic/cmake-build-debug/src/mosaic && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Mosaic_lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/mosaic/CMakeFiles/Mosaic_lib.dir/build: src/mosaic/libMosaic_lib.a

.PHONY : src/mosaic/CMakeFiles/Mosaic_lib.dir/build

src/mosaic/CMakeFiles/Mosaic_lib.dir/clean:
	cd /Users/normadi/Downloads/mosaic/cmake-build-debug/src/mosaic && $(CMAKE_COMMAND) -P CMakeFiles/Mosaic_lib.dir/cmake_clean.cmake
.PHONY : src/mosaic/CMakeFiles/Mosaic_lib.dir/clean

src/mosaic/CMakeFiles/Mosaic_lib.dir/depend:
	cd /Users/normadi/Downloads/mosaic/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/normadi/Downloads/mosaic /Users/normadi/Downloads/mosaic/src/mosaic /Users/normadi/Downloads/mosaic/cmake-build-debug /Users/normadi/Downloads/mosaic/cmake-build-debug/src/mosaic /Users/normadi/Downloads/mosaic/cmake-build-debug/src/mosaic/CMakeFiles/Mosaic_lib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/mosaic/CMakeFiles/Mosaic_lib.dir/depend

