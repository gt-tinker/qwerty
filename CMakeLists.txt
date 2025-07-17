cmake_minimum_required(VERSION 3.15)

project(qwerty_mlir LANGUAGES C CXX)

option(DUMP_MLIR_DEPS "Generate mlir-deps.tsv. Useful for linking without CMake" OFF)

# These flags are used if cmake is called with -DCMAKE_BUILD_TYPE=Coverage
# https://gcovr.com/en/stable/cookbook.html#oos-cmake
list(APPEND CMAKE_CONFIGURATION_TYPES Coverage)
set(CMAKE_CXX_FLAGS_COVERAGE "-g -O0 --coverage")
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE --coverage)
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE --coverage)

# Need to use C++17 because mockturtle contains some syntax that is invalid in
# C++20. This has been fixed (https://github.com/lsils/mockturtle/pull/488),
# but the vendored version of mockturtle inside the tweedledum repository has
# not yet been updated to include this fix
set(CMAKE_CXX_STANDARD 17)

# Fyi, a debug build causes linker errors on Windows:
# https://stackoverflow.com/a/48952485/321301
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # Hide massive amount of MSVC warnings
    add_compile_options(/w)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Don't bother showing warnings on macOS. Clang is too noisy
    add_compile_options(-Wall -Wextra -pedantic -fdiagnostics-color)
endif()

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(ExternalProject)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/qwerty_util/)
include_directories(${PROJECT_SOURCE_DIR}/qwerty_mlir/include)
include_directories(${PROJECT_BINARY_DIR}/qwerty_mlir/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Used for tweedledum below
# From: https://github.com/Microsoft/vcpkg/issues/3489#issuecomment-469691199
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(STATIC_LIBRARY_PREFIX "")
    set(STATIC_LIBRARY_SUFFIX ".lib")
else()
    set(STATIC_LIBRARY_PREFIX "lib")
    set(STATIC_LIBRARY_SUFFIX ".a")
endif()

# This is to make Tweedledum happy below. Without this, it wants to install to
# /usr/local. We can't disable Tweedledum installing entirely because a
# consumer of this library (qwerty_mlir+qwutil) will need the static tweedledum
# libs to link.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set_property(CACHE CMAKE_INSTALL_PREFIX PROPERTY VALUE "${CMAKE_BINARY_DIR}/install-dir")
endif()

# Tweedledum
ExternalProject_Add(tweedledum
    # Use a submodule because if we have CMake clone the git url (i.e., provide
    # GIT_REPOSITORY here), CMake will unnecessarily re-clone it for every
    # invocation of cmake
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/tweedledum
    # Last part is because tweedledum is seemingly built without -fPIC by
    # default despite compiling to a library. Possible MSVC++-ism?
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DTWEEDLEDUM_PYBINDS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    # EXTREME hack: For some reason, in its infinite wisdom, setuptools(?) uses
    # an ever-changing symlink to ninja (i.e., the path to ninja actually
    # changes for every invocation of `pip install'). So to spare ourselves the
    # grief of trying to monkeypatch the ninja path in CMakeCache.txt, just
    # delete the CMake cache entirely
    #PATCH_COMMAND cmake -E rm -f -- <BINARY_DIR>/CMakeCache.txt
    # Needed for Ninja to know to build this before linking _qwerty_harness
    BUILD_BYPRODUCTS <BINARY_DIR>/${STATIC_LIBRARY_PREFIX}tweedledum${STATIC_LIBRARY_SUFFIX}
)
ExternalProject_Get_Property(tweedledum BINARY_DIR)
set(TWEEDLEDUM_LIB_PATH ${BINARY_DIR}/${STATIC_LIBRARY_PREFIX}tweedledum${STATIC_LIBRARY_SUFFIX})
ExternalProject_Get_Property(tweedledum SOURCE_DIR)
set(TWEEDLEDUM_INCLUDE_PATH
    ${SOURCE_DIR}/include
    # TODO: Figure out a way not to have to hardcode all of tweedledum's
    #       vendored dependencies. Perhaps installing it to a dummy path would
    #       help?
    ${SOURCE_DIR}/external/mockturtle
    ${SOURCE_DIR}/external/kitty
    ${SOURCE_DIR}/external/parallel_hashmap
    ${SOURCE_DIR}/external/eigen
    ${SOURCE_DIR}/external/fmt/include
    ${SOURCE_DIR}/external/nlohmann
)
# Use SYSTEM to avoid spamming output with warnings from mockturtle/tweedledum headers
include_directories(SYSTEM ${TWEEDLEDUM_INCLUDE_PATH})
# Needed to avoid linker errors related to tweedledum's {fmt} dependency
add_compile_definitions(FMT_HEADER_ONLY=1)

add_subdirectory(qwerty_mlir)

# tweedledum.cpp is used by both libqwc and QwertyToQCircConversionPass. We
# separate it into its own library libqwutil to avoid a dependency cycle
add_library(qwutil STATIC qwerty_util/tweedledum.cpp)
target_link_libraries(qwutil ${TWEEDLEDUM_LIB_PATH})
# Need tablegen-generated header files
add_dependencies(qwutil MLIRQCircOpsIncGen)

install(TARGETS qwutil)

# This is a degenerate kludge needed by qwerty-mlir-sys. Although LLVM has
# llvm-config to tell downstream users the order needed for linking LLVM static
# libs, MLIR has nothing like this at the time of writing; instead, MLIR
# downstream users are expected to rely on CMake. However, we cannot ask CMake
# for the right ordering for MLIR static libraries in the build.rs in
# qwerty-mlir-sys. Instead, we can dump the adjacency list for the MLIR
# dependency graph to a text file for build.rs to read.
if (DUMP_MLIR_DEPS)
    set(dep_file_contents "")
    get_property(imported_targets DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                 PROPERTY IMPORTED_TARGETS)

    foreach(tgt ${imported_targets})
        string(FIND ${tgt} "MLIR" pos)

        if (pos EQUAL 0)
            string(APPEND dep_file_contents "${tgt}")

            get_property(deps TARGET ${tgt}
                         PROPERTY INTERFACE_LINK_LIBRARIES)
            foreach(dep ${deps})
                string(FIND ${dep} "MLIR" pos)

                if (pos EQUAL 0)
                    string(APPEND dep_file_contents "\t${dep}")
                endif()
            endforeach()

            string(APPEND dep_file_contents "\n")
        endif()
    endforeach()

    set(dep_file_path "${CMAKE_BINARY_DIR}/mlir-deps.tsv")
    file(WRITE ${dep_file_path} ${dep_file_contents})
    install(FILES ${dep_file_path} TYPE LIB)
endif()
