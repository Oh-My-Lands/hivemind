cmake_minimum_required(VERSION 3.16)
project(hivemind)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 23)

# Performance flags:
#   -Ofast: Maximum optimization (includes -O3 + fast-math)
#   -flto: Link-time optimization for cross-file inlining
#
# HIVEMIND_ARCH controls -march. It defaults to native, which is right when the
# machine that builds is the machine that runs -- a dev box, or a pod where you
# compile in place.
#
# It is wrong for a container image, because the build host and the run host are
# different machines. -march=native on a Tiger Lake laptop emits AVX-512; a
# RunPod serverless worker on EPYC Zen 3 has no AVX-512, so the binary takes
# SIGILL on the first such instruction. That happens before main() prints
# anything, so the only visible symptom is a worker that produces no output and
# exits rc=-4 -- which looks like a hang, not a crash.
#
# Container builds must pass an explicit baseline. x86-64-v3 (AVX2, FMA, BMI2,
# F16C, LZCNT) is supported by everything since Haswell 2013 / Zen 1 2017, so it
# covers any GPU host worth renting while keeping the vectorisation that matters.
set(HIVEMIND_ARCH "native" CACHE STRING "Target CPU architecture for -march")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -DNNUE_EMBEDDING_OFF -march=${HIVEMIND_ARCH} -flto")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")

# Add source files
set(LIB_SOURCES
    src/engine.cc
    src/board.cc
    src/node.cc
    src/planes.cc
    src/searchthread.cc
    src/uci.cc
    src/agent.cc
    src/onnx_utils.cc
    src/benchmark.cc
    src/zobrist.cc
    src/globals.cc
    src/rl/gamepgn.cc
    src/rl/selfplay.cc
    src/rl/training_data_writer.cc
    src/rl/model_eval.cc
)

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(TensorRT_DIR /usr/src/tensorrt/)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

find_package(Threads REQUIRED)
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)

add_subdirectory(src/Fairy-Stockfish)

add_library(${PROJECT_NAME}_lib STATIC ${LIB_SOURCES})
target_include_directories(${PROJECT_NAME}_lib PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES} Fairy-Stockfish Threads::Threads)

add_executable(${PROJECT_NAME} src/main.cc)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_lib)

include(CTest)
enable_testing()
add_subdirectory(tests)