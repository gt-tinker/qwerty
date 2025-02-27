# Based on the following (but slowly diverging):
# https://github.com/scikit-build/scikit-build-sample-projects/blob/52f172abac47798dbedcdf2a35d6e2833f2c1ad2/projects/hello-cpp/CMakeLists.txt

cmake_minimum_required(VERSION 3.15...3.27)

project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

option(BUILD_TESTS "Build tests" OFF)
option(USE_QIREE "Enable QIREE integration" OFF)

# These flags are used if cmake is called with -DCMAKE_BUILD_TYPE=Coverage
# https://gcovr.com/en/stable/cookbook.html#oos-cmake
list(APPEND CMAKE_CONFIGURATION_TYPES Coverage)
set(CMAKE_CXX_FLAGS_COVERAGE "-g -O0 --coverage")
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE --coverage)
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE --coverage)

# TODO: Remove both of these
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Extract this part of the line:
    #     GNU gold (GNU Binutils for Debian 2.40) 1.16
    #                                       ^^^^
    # or, on RHEL:
    #     GNU gold (version 2.30-125.el8_10) 1.15
    #                       ^^^^
    execute_process(COMMAND /bin/bash -c "ld.gold --version | head -1 | cut -d ')' -f 1 | awk '{print $NF}' | cut -d - -f 1" OUTPUT_VARIABLE GOLD_VERSION)

    # --dependency-file was added to gold in f37b21b481a780, first seen in
    # version 2.35 of binutils. For some reason, CMake decides to emit an ld
    # line that includes --dependency-file, so we should not use gold unless it
    # is new enough to support this
    if (GOLD_VERSION VERSION_GREATER_EQUAL "2.35")
        set(CMAKE_CXX_FLAGS "-fuse-ld=gold")
    endif()
endif()

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
include_directories(${PROJECT_SOURCE_DIR}/qwerty/)
include_directories(${PROJECT_SOURCE_DIR}/qwerty_mlir/include)
include_directories(${PROJECT_BINARY_DIR}/qwerty_mlir/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

if (USE_QIREE)
    find_package(QIREE REQUIRED CONFIG)
    add_compile_definitions(QWERTY_USE_QIREE=1)
    # God forgive me for this. https://stackoverflow.com/q/43551483/321301
    set(CMAKE_MACOSX_RPATH 1)
    list(APPEND CMAKE_INSTALL_RPATH ${XACC_LIBRARY_DIR})
endif()

# Used for tweedledum and qir-runner below
# From: https://github.com/Microsoft/vcpkg/issues/3489#issuecomment-469691199
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(STATIC_LIBRARY_PREFIX "")
    set(STATIC_LIBRARY_SUFFIX ".lib")
else()
    set(STATIC_LIBRARY_PREFIX "lib")
    set(STATIC_LIBRARY_SUFFIX ".a")
endif()

if (BUILD_TESTS)
    ExternalProject_Add(gtest
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/tpls/googletest
        # -Dgtest_force_shared_crt=ON is to make the Windows linker happy:
        # https://stackoverflow.com/a/28887127/321301
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_GMOCK=OFF -Dgtest_force_shared_crt=ON
        INSTALL_COMMAND cmake -E echo "Skipping install step as we only need the static gtest libraries"
        BUILD_BYPRODUCTS <BINARY_DIR>/lib/${STATIC_LIBRARY_PREFIX}gtest${STATIC_LIBRARY_SUFFIX}
    )
    ExternalProject_Get_Property(gtest SOURCE_DIR)
    set(GTEST_INCLUDE_PATH ${SOURCE_DIR}/googletest/include)
    ExternalProject_Get_Property(gtest BINARY_DIR)
    set(GTEST_LIB_PATH ${BINARY_DIR}/lib/${STATIC_LIBRARY_PREFIX}gtest${STATIC_LIBRARY_SUFFIX})

    # Create CTestTestfile.cmake at the root here so that `ctest'ing devs do
    # not need to go on a scavenger hunt to find it deep in
    # _skbuild/test/qwerty/
    enable_testing()
    add_compile_definitions(QWERTY_INCLUDE_TESTS=1)
endif()

# Tweedledum
ExternalProject_Add(tweedledum
    # Use a submodule because if we have CMake clone the git url (i.e., provide
    # GIT_REPOSITORY here), CMake will unnecessarily re-clone it for every
    # invocation of cmake
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/tpls/tweedledum
    # Last part is because tweedledum is seemingly built without -fPIC by
    # default despite compiling to a library. Possible MSVC++-ism?
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DTWEEDLEDUM_PYBINDS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    # EXTREME hack: For some reason, in its infinite wisdom, setuptools(?) uses
    # an ever-changing symlink to ninja (i.e., the path to ninja actually
    # changes for every invocation of `pip install'). So to spare ourselves the
    # grief of trying to monkeypatch the ninja path in CMakeCache.txt, just
    # delete the CMake cache entirely
    #PATCH_COMMAND cmake -E rm -f -- <BINARY_DIR>/CMakeCache.txt
    # The tweedledum build files try to install to /usr/local, which is not
    # necessary. Fix taken from https://stackoverflow.com/a/39811200/321301
    INSTALL_COMMAND cmake -E echo "Skipping install step as we only need the static tweedledum library"
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

# QIR Runner
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(RUST_BUILD_TYPE dev)
    # Not sure why, but passing `--profile dev' to cargo puts the binaries in
    # `target/debug' instead of `target/dev'
    set(RUST_BUILD_TARGET_DIR debug)
else() # Release
    set(RUST_BUILD_TYPE release)
    set(RUST_BUILD_TARGET_DIR release)
endif()
ExternalProject_Add(qir-runner
    # Use a submodule because if we have CMake clone the git url (i.e., provide
    # GIT_REPOSITORY here), CMake will unnecessarily re-clone it for every
    # invocation of cmake
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/tpls/qir-runner
    CONFIGURE_COMMAND cmake -E echo "Skipping configure step as we do not need it for a Rust build"
    BUILD_COMMAND cargo build --package qir-backend --profile ${RUST_BUILD_TYPE} --target-dir <BINARY_DIR> --manifest-path <SOURCE_DIR>/Cargo.toml
    INSTALL_COMMAND cmake -E echo "Skipping install step as we only need the static qir-runner libraries"
    # Needed for Ninja to know to build this before linking _qwerty_harness
    BUILD_BYPRODUCTS <BINARY_DIR>/${RUST_BUILD_TARGET_DIR}/${STATIC_LIBRARY_PREFIX}qir_backend${STATIC_LIBRARY_SUFFIX}
)
ExternalProject_Get_Property(qir-runner BINARY_DIR)
set(QIR_RUNNER_LIB_PATH ${BINARY_DIR}/${RUST_BUILD_TARGET_DIR}/${STATIC_LIBRARY_PREFIX}qir_backend${STATIC_LIBRARY_SUFFIX})

# qir-runner apparently needs this
# From: https://stackoverflow.com/a/18559081/321301
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # Found these by running
    #     cargo rustc -- --print=native-static-libs
    # in tpls\qir-runner\backend.
    # Found from (unrelated thread): https://github.com/corrosion-rs/corrosion/issues/418#issuecomment-1636381027
    set(winlibs bcrypt kernel32 advapi32 userenv kernel32 kernel32 ws2_32 bcrypt ntdll msvcrt legacy_stdio_definitions)
else()
    set(winlibs "")
endif()

if (USE_QIREE)
    set(qiree_libs QIREE::qirxacc QIREE::qiree)
else()
    set(qiree_libs "")
endif()

# This is a bit ugly, but the goal is just to set ${all_libs} to contain pretty
# much the whole dependency closure of the Qwerty compiler. This is used in
# both building qwerty-test and also _qwerty_harness.so. However, because we
# add_subdirectory(test) below, and that actually adds some test passes, then
# we can't do this before test/CMakeLists.txt runs. Yet we also cannot do it
# after because qwerty-test needs ${all_libs}! So we resort to this macro to
# break the cycle
macro(calc_all_libs)
    llvm_map_components_to_libnames(llvm_libs support core native)
    get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
    set(more_mlir_libs MLIRExecutionEngine)
    set(all_libs ${llvm_libs} ${dialect_libs} ${more_mlir_libs} ${winlibs} ${QIR_RUNNER_LIB_PATH})
endmacro()

add_subdirectory(qwerty_mlir)
if (BUILD_TESTS)
    add_subdirectory(test)
endif()

add_library(qwc STATIC qwerty/ast.cpp
                       qwerty/classical_kernel.cpp
                       qwerty/mlir_handle.cpp
                       qwerty/qpu_kernel.cpp
                       qwerty/ast_visitor/flag_immat.cpp
                       qwerty/ast_visitor/canonicalize.cpp
                       qwerty/ast_visitor/desugar.cpp
                       qwerty/ast_visitor/graphviz.cpp
                       qwerty/ast_visitor/find_instants.cpp
                       qwerty/ast_visitor/eval_dimvar_exprs.cpp
                       qwerty/ast_visitor/type_checking.cpp
                       qwerty/ast_visitor/classical_netlist.cpp
                       qwerty/ast_visitor/qpu_lowering.cpp)
target_link_libraries(qwc ${qiree_libs})
# Need tablegen-generated header files
add_dependencies(qwc MLIRQCircOpsIncGen
                     MLIRQCircPassIncGen
                     MLIRQwertyOpsIncGen
                     MLIRQwertyPassIncGen)

# tweedledum.cpp is used by both libqwc and QwertyToQCircConversionPass. We
# separate it into its own library libqwutil to avoid a dependency cycle
add_library(qwutil STATIC qwerty/tweedledum.cpp)
target_link_libraries(qwutil ${TWEEDLEDUM_LIB_PATH})
# Need tablegen-generated header files
add_dependencies(qwutil MLIRQCircOpsIncGen)

# Now, for the Python extension
calc_all_libs()
find_package(Python COMPONENTS Interpreter Development.SABIModule REQUIRED)
Python_add_library(_qwerty_harness MODULE
                   qwerty/_qwerty_harness.cpp
                   WITH_SOABI USE_SABI 3.10)
target_link_libraries(_qwerty_harness PRIVATE qwc ${all_libs})
install(TARGETS _qwerty_harness DESTINATION ${SKBUILD_PROJECT_NAME})
