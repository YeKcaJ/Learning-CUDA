cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
project(low_precision_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
  find_program(SHA256SUM_EXECUTABLE NAMES sha256sum REQUIRED)
  add_test(NAME core_tools COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tools.py)
endif()

# 同一套核心源码编译为两个独立程序，格式在编译时确定。
foreach(FORMAT IN ITEMS mxfp8 nvfp4)
  set(TARGET pipeline_${FORMAT})
  add_executable(${TARGET} pipeline_main.cu tests/reference.cpp)
  target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  if(FORMAT STREQUAL "nvfp4")
    target_compile_definitions(${TARGET} PRIVATE TEST_NVFP4)
  endif()

  if(BUILD_TESTING)
    string(TOUPPER "${FORMAT}" UPPER_FORMAT)
    set(CPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../CPU${UPPER_FORMAT}")
    add_test(NAME ${FORMAT}_pipeline_io
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_pipeline.py
              ${FORMAT} $<TARGET_FILE:${TARGET}>)
    add_test(NAME ${FORMAT}_pipeline_regression
      COMMAND ${TARGET} --self-test ${CPU_DIR})
    add_test(NAME ${FORMAT}_frozen_hashes
      COMMAND ${SHA256SUM_EXECUTABLE} -c tests/golden/SHA256SUMS.txt)
    set_tests_properties(${FORMAT}_frozen_hashes PROPERTIES WORKING_DIRECTORY "${CPU_DIR}")
  endif()
endforeach()
