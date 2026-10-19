cmake_minimum_required(VERSION 3.18)
project(lobsim LANGUAGES CXX)

option(LOB_USE_CUDA "Build the CUDA agent back end" OFF)
option(LOB_USE_MPI "Build with MPI (instruments split across ranks)" OFF)
option(LOB_NO_PAD "Remove cache-line padding of per-instrument counters (false-sharing experiment)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

if(LOB_USE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)  # Tesla T4; override with -DCMAKE_CUDA_ARCHITECTURES=86 etc.
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  find_package(CUDAToolkit REQUIRED)
endif()

find_package(OpenMP REQUIRED)

add_library(lobcore src/order_book.cpp src/sim.cpp)
target_include_directories(lobcore PUBLIC include)
target_link_libraries(lobcore PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(lobcore PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
if(LOB_NO_PAD)
  target_compile_definitions(lobcore PUBLIC LOB_NO_PAD)
endif()

if(LOB_USE_CUDA)
  target_sources(lobcore PRIVATE src/decide_cuda.cu)
  target_compile_definitions(lobcore PUBLIC LOB_USE_CUDA)
  # -lineinfo lets Nsight Compute map metrics back to source lines.
  target_compile_options(lobcore PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -O3>)
  target_link_libraries(lobcore PUBLIC CUDA::cudart)
endif()

if(LOB_USE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_compile_definitions(lobcore PUBLIC LOB_USE_MPI)
  target_link_libraries(lobcore PUBLIC MPI::MPI_CXX)
endif()

add_executable(lobsim src/main.cpp)
target_link_libraries(lobsim PRIVATE lobcore)

enable_testing()
add_executable(lob_tests tests/test_main.cpp)
target_link_libraries(lob_tests PRIVATE lobcore)
add_test(NAME unit_tests COMMAND lob_tests)
