cmake_minimum_required(VERSION 3.15)

project(qwerty_mlir LANGUAGES C CXX)

option(DUMP_MLIR_DEPS "Generate mlir-deps.tsv. Useful for linking without CMake" OFF)

# These flags are used if cmake is called with -DCMAKE_BUILD_TYPE=Coverage
# https://gcovr.com/en/stable/cookbook.html#oos-cmake
list(APPEND CMAKE_CONFIGURATION_TYPES Coverage)
set(CMAKE_CXX_FLAGS_COVERAGE "-g -O0 --coverage")
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE --coverage)
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE --coverage)

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

include_directories(${PROJECT_SOURCE_DIR}/eigen)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/qwerty_mlir/include)
include_directories(${PROJECT_BINARY_DIR}/qwerty_mlir/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(qwerty_mlir)

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
