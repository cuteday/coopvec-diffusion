include(FetchContent)

if (NOT WIN32)
  message(FATAL_ERROR "TensorRT-RTX download is only supported on Windows.")
endif()

if (NOT TRT_RTX_URL)
  set(TRT_RTX_URL "https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.2/tensorrt-rtx-1.2.0.54-win10-amd64-cuda-12.9-release-external.zip")
endif()

if (NOT TRT_RTX_PATH)
  set(TRT_RTX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/trt-rtx" CACHE STRING "" FORCE)
endif()

FetchContent_Declare(
  tensorrt_rtx
  URL "${TRT_RTX_URL}"
  SOURCE_DIR "${TRT_RTX_PATH}"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
message(STATUS "Fetching TensorRT-RTX from ${TRT_RTX_URL} into ${TRT_RTX_PATH}")

FetchContent_MakeAvailable(tensorrt_rtx)

set(TRT_RTX_ROOT "${tensorrt_rtx_SOURCE_DIR}" CACHE STRING "Path to the TensorRT-RTX root directory" FORCE)
set(TRT_RTX_INCLUDE_DIR "${TRT_RTX_ROOT}/include" CACHE STRING "Path to the TensorRT-RTX include directory" FORCE)
set(TRT_RTX_LIBRARY_DIR "${TRT_RTX_ROOT}/lib" CACHE STRING "Path to the TensorRT-RTX library directory" FORCE)

# Read TensorRT-RTX version info directly from NvInferVersion.h
set(TRT_RTX_VERSION_HEADER "${TRT_RTX_INCLUDE_DIR}/NvInferVersion.h")
if(NOT EXISTS "${TRT_RTX_VERSION_HEADER}")
  message(FATAL_ERROR "TensorRT-RTX version header not found: ${TRT_RTX_VERSION_HEADER}")
endif()

file(STRINGS "${TRT_RTX_VERSION_HEADER}" TRT_RTX_VERSION_STRINGS REGEX "#define TRT_.*_RTX")
if(NOT TRT_RTX_VERSION_STRINGS)
  message(FATAL_ERROR "No TRT_*_RTX version defines found in ${TRT_RTX_VERSION_HEADER}, please check if the path provided is correct.")
endif()

foreach(type MAJOR MINOR PATCH)
  set(trt_${type} "")
  foreach(version_line ${TRT_RTX_VERSION_STRINGS})
    string(REGEX MATCH "TRT_${type}_RTX [0-9]+" trt_type_string "${version_line}")
    if(trt_type_string)
      string(REGEX MATCH "[0-9]+" trt_${type} "${trt_type_string}")
      break()
    endif()
  endforeach()
  if(NOT DEFINED trt_${type})
    message(FATAL_ERROR "Failed to extract TRT_${type}_RTX from ${TRT_RTX_VERSION_HEADER}")
  endif()
endforeach()

set(TRT_RTX_VERSION "${trt_MAJOR}.${trt_MINOR}.${trt_PATCH}" CACHE STRING "TensorRT-RTX version string" FORCE)
set(TRT_RTX_SOVERSION "${trt_MAJOR}_${trt_MINOR}" CACHE STRING "TensorRT-RTX SOVERSION string" FORCE)

# Derive library names from the SOVERSION
set(_trtrtx_lib_base "tensorrt_rtx")
set(_trtrtx_onnxparser_lib_base "tensorrt_onnxparser_rtx")
if(WIN32)
  set(TRT_RTX_LIB_NAME "${_trtrtx_lib_base}_${TRT_RTX_SOVERSION}" CACHE STRING "TensorRT-RTX library name" FORCE)
  set(TRT_RTX_ONNXPARSER_LIB_NAME "${_trtrtx_onnxparser_lib_base}_${TRT_RTX_SOVERSION}" CACHE STRING "TensorRT-ONNXParser-RTX library name" FORCE)
else()
  set(TRT_RTX_LIB_NAME "${_trtrtx_lib_base}" CACHE STRING "TensorRT-RTX library name" FORCE)
  set(TRT_RTX_ONNXPARSER_LIB_NAME "${_trtrtx_onnxparser_lib_base}" CACHE STRING "TensorRT-ONNXParser-RTX library name" FORCE)
endif()
