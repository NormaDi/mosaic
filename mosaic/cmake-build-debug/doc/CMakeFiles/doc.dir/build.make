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

# Utility rule file for doc.

# Include the progress variables for this target.
include doc/CMakeFiles/doc.dir/progress.make

doc/CMakeFiles/doc:
	cd /Users/normadi/Downloads/mosaic && /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E echo_append Building\ Documentation...
	cd /Users/normadi/Downloads/mosaic && /usr/local/bin/doxygen /Users/normadi/Downloads/mosaic/cmake-build-debug/doc/doxyfile
	cd /Users/normadi/Downloads/mosaic && /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E echo Done.

doc: doc/CMakeFiles/doc
doc: doc/CMakeFiles/doc.dir/build.make

.PHONY : doc

# Rule to build all files generated by this target.
doc/CMakeFiles/doc.dir/build: doc

.PHONY : doc/CMakeFiles/doc.dir/build

doc/CMakeFiles/doc.dir/clean:
	cd /Users/normadi/Downloads/mosaic/cmake-build-debug/doc && $(CMAKE_COMMAND) -P CMakeFiles/doc.dir/cmake_clean.cmake
.PHONY : doc/CMakeFiles/doc.dir/clean

doc/CMakeFiles/doc.dir/depend:
	cd /Users/normadi/Downloads/mosaic/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/normadi/Downloads/mosaic /Users/normadi/Downloads/mosaic/doc /Users/normadi/Downloads/mosaic/cmake-build-debug /Users/normadi/Downloads/mosaic/cmake-build-debug/doc /Users/normadi/Downloads/mosaic/cmake-build-debug/doc/CMakeFiles/doc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : doc/CMakeFiles/doc.dir/depend
