cmake_minimum_required(VERSION 3.18)

project(
	Fluxel
	DESCRIPTION "Implementing simple diffusion models using cooperative vectors. This is my pet project named Fluxel."
	LANGUAGES CXX
)

option(FLUXEL_WITH_CUDA "Enable CUDA support" ON)

set(FLUXEL_PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "The root directory of the Fluxel project.")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_ENABLE_EXTENDED_ALIGNED_STORAGE ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819 /wd4068 /wd4828")	# chn encoding warning

add_compile_definitions ("$<$<CONFIG:Debug>:FLUXEL_DEBUG_BUILD>")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/common/cmake")

if (MSVC)
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_ITERATOR_DEBUG_LEVEL=1")
endif()

set(FLUXEL_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/bin")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${FLUXEL_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${FLUXEL_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${FLUXEL_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${FLUXEL_BINARY_DIR}")

# Donut and ShaderMake related options
option(DONUT_WITH_DX11 "Cooperative vector does not support DX11." OFF)
option(DONUT_WITH_DX12 "Enabled the DXIL path for DX12 support (windows only)." OFF)
option(DONUT_WITH_VULKAN "Enabled the SPIR-V path for Vulkan support." ON)

set(SHADERMAKE_BIN_OUTPUT_PATH "${FLUXEL_BINARY_DIR}/ShaderMake")
# We need newer DXC version than below for cooperative vectors support
set(SHADERMAKE_DXC_VERSION "v1.8.2505" CACHE STRING "")
set(SHADERMAKE_DXC_DATE "2025_05_24" CACHE STRING "")
option(SHADERMAKE_FIND_SLANG "" ON)
set(SHADERMAKE_SLANG_VERSION "2025.10" CACHE STRING "")

set(DONUT_SHADERS_OUTPUT_DIR "${FLUXEL_BINARY_DIR}/shaders/framework")

if (WIN32)
	# DX12 is enabled only on Windows with a GPU&driver that supports it
	set(DONUT_WITH_DX12 ON)
	set(NVRHI_WITH_DX12 ON)
endif()

if (DONUT_WITH_DX12)
	# Get D3D Agility SDK Preview for Coop Vector support
	set(D3D_AGILITY_SDK_PREVIEW_VERSION "1.717.0-preview")
	set(DONUT_D3D_AGILITY_SDK_URL "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/${D3D_AGILITY_SDK_PREVIEW_VERSION}")
	set(DONUT_D3D_AGILITY_SDK_FETCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/dx12-agility-sdk" CACHE STRING "" FORCE)
	include("${CMAKE_CURRENT_SOURCE_DIR}/external/donut/cmake/FetchAgilitySDK.cmake")
	include("${CMAKE_CURRENT_SOURCE_DIR}/common/cmake/ConfigureAgilitySDK.cmake")
endif()

if (FLUXEL_WITH_CUDA)
	find_package(CUDAToolkit REQUIRED)
	include("${CMAKE_CURRENT_SOURCE_DIR}/common/cmake/FetchTensorRTRTX.cmake")
	include_directories(${TRT_RTX_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
	link_directories(${TRT_RTX_LIBRARY_DIR} ${CUDA_LIBRARY_DIRS})
endif()

add_subdirectory(external/donut)
add_subdirectory(src)
add_subdirectory(samples)

if (NOT EXISTS "${SHADERMAKE_SLANG_PATH}")
	message(FATAL_ERROR "SHADERMAKE_SLANG_PATH not set!")
endif()

if (WIN32)
	cmake_path(REMOVE_FILENAME SHADERMAKE_SLANG_PATH OUTPUT_VARIABLE SLANG_DIRECTORY)
	cmake_path(REPLACE_FILENAME SHADERMAKE_DXC_PATH "dxcompiler.dll" OUTPUT_VARIABLE DXCOMPILER_DLL)
	cmake_path(REPLACE_FILENAME SHADERMAKE_DXC_PATH "dxil.dll" OUTPUT_VARIABLE DXIL_DLL)
	file(COPY "${DXCOMPILER_DLL}" "${DXIL_DLL}" DESTINATION "${SLANG_DIRECTORY}/")
endif()
