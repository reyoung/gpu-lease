cmake_minimum_required(VERSION 3.18)

project(gpu_lease VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(gpu_lease_core STATIC
  src/gpu_lease/app.cpp
  src/gpu_lease/busy.cpp
  src/gpu_lease/client.cpp
  src/gpu_lease/common.cpp
  src/gpu_lease/gpu_state.cpp
  src/gpu_lease/manager.cpp
  src/gpu_lease/server.cpp
)
target_include_directories(gpu_lease_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(gpu_lease_core PUBLIC cxx_std_17)
target_link_libraries(gpu_lease_core PUBLIC
  CUDA::cublas
  CUDA::cudart
  Threads::Threads
  ${CMAKE_DL_LIBS}
)

add_executable(gpu-lease src/main.cpp)
target_link_libraries(gpu-lease PRIVATE gpu_lease_core)

enable_testing()

add_executable(gpu_lease_tests tests/gpu_lease_tests.cpp)
target_link_libraries(gpu_lease_tests PRIVATE gpu_lease_core)
add_test(NAME gpu_lease_tests COMMAND gpu_lease_tests)
