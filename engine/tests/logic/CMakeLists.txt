cmake_minimum_required(VERSION 3.16)
project(hivemind_logic_tests CXX)

# A separate CMake root, not a subdirectory of engine/CMakeLists.txt, and
# deliberately so.
#
# The parent does find_package(TensorRT REQUIRED) and find_package(CUDA
# REQUIRED) before it reaches any test target, and engine/tests links
# hivemind_lib, which pulls in engine.cc. So the existing suite cannot even be
# *configured* without a GPU toolchain -- on a machine with no nvcc there is no
# way to run a single assertion.
#
# The logic under test here is pure arithmetic over four integers. Keeping it in
# a root that depends on nothing means the sign conventions, the diagonal
# pairing and the flag semantics -- the parts most likely to be subtly wrong and
# most expensive to debug on rented hardware -- can be checked on a laptop.
#
#   cmake -S engine/tests/logic -B build-logic
#   cmake --build build-logic
#   ctest --test-dir build-logic --output-on-failure

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# No -Ofast/-march=native here. The parent sets those for the engine, where they
# matter; for assertions over integers they only risk making the test binary
# behave unlike a debug build.
add_compile_options(-Wall -Wextra)

# Required before gtest_discover_tests registers anything. The sibling suite in
# engine/tests gets this from the top-level project; a standalone root does not,
# and without it the binary builds and ctest reports "No tests were found".
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(hivemind_logic_tests test_time_control.cc)
target_include_directories(hivemind_logic_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(hivemind_logic_tests PRIVATE gtest_main)

include(GoogleTest)
gtest_discover_tests(hivemind_logic_tests)
