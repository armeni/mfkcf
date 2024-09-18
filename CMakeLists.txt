cmake_minimum_required(VERSION 3.16)
set(CMAKE_HAVE_LIBC_PTHREAD 1)

option(USE_CUDA "enable ONNXRuntime engine with CUDA provider" ON)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-fPIC -std=c++17")
set(CMAKE_C_FLAGS "-fPIC -std=c11  -O3") # -mavx512f
set(CMAKE_BUILD_TYPE "Release")
project(mixformer-kcf)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.0)
find_package(CUDA REQUIRED)


if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("--embedded platform on")

  # cuda
  include_directories(/usr/local/cuda-12.0/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda-12.0/targets/aarch64-linux/lib)

  # tensorrt
  include_directories(/usr/include/aarch64-linux-gnu)
  link_libraries(/usr/lib/aarch64-linux-gnu)
else()
  message("--embedded platform off")
  
  # cuda
  include_directories(/usr/local/cuda-12.0/include)
  link_directories(/usr/local/cuda-12.0/lib64)

  # tensorrt
  include_directories(/opt/TensorRT/include)
  link_directories(/opt/TensorRT/lib)

endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(/usr/include/opencv4)
include_directories(/home/uavlab20/onnxruntime)

link_directories(/usr/lib/x86_64-linux-gnu /home/uavlab20/onnxruntime/build/Linux/Release)

add_executable(tracker tracking.cpp utils.cpp mixformer/mixformer_trt.cpp mixformer/logging.h mixformer/half.hpp kcf/kcftracker.cpp kcf/kcftracker.hpp kcf/fhog.hpp kcf/fhog.cpp kcf/ffttools.hpp kcf/labdata.hpp)
target_link_libraries(tracker nvinfer cudart libopencv_core.so libopencv_highgui.so libopencv_imgproc.so libopencv_video.so libopencv_videoio.so libopencv_imgcodecs.so)
