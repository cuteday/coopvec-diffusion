set(LIBRARY_FILTER "src")

# Collect sources per folder so they can be toggled independently
set(SRC_FILES "")

file(GLOB_RECURSE CORE_SRC "Core/*.cpp" "Core/*.h")
list(APPEND SRC_FILES ${CORE_SRC})

file(GLOB_RECURSE NEURAL_SRC "Neural/*.cpp" "Neural/*.h")
list(APPEND SRC_FILES ${NEURAL_SRC})

file(GLOB_RECURSE UTILS_SRC "Utils/*.cpp" "Utils/*.h")
list(APPEND SRC_FILES ${UTILS_SRC})

# TensorRT sources are only used when CUDA is enabled
if (FLUXEL_WITH_CUDA)
    file(GLOB_RECURSE TENSORRT_SRC "TensorRT/*.cpp" "TensorRT/*.h")
    list(APPEND SRC_FILES ${TENSORRT_SRC})
endif()

set(project "FluxelLib")
set(folder "${LIBRARY_FILTER}")

add_subdirectory(Core/Math)
add_subdirectory(Shaders)

add_library(${project} STATIC EXCLUDE_FROM_ALL ${SRC_FILES})
target_include_directories(${project} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${project} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Core)
target_link_libraries(${project} PUBLIC donut_app donut_engine krr_math)

if (DONUT_WITH_DX12)
    add_dependencies(${project} dx12-agility-sdk)
    target_compile_definitions(
		${project} PRIVATE DONUT_D3D_AGILITY_SDK_VERSION=${DONUT_D3D_AGILITY_SDK_VERSION}
		${project} PRIVATE DONUT_D3D_AGILITY_PREVIEW_SDK_VERSION=${DONUT_D3D_AGILITY_PREVIEW_SDK_VERSION}
	)
endif()

if (FLUXEL_WITH_CUDA)
    target_include_directories(${project} PUBLIC ${TRT_RTX_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${project} PUBLIC ${TRT_RTX_LIB_NAME} ${TRT_RTX_ONNXPARSER_LIB_NAME} ${CUDA_LIBRARIES} CUDA::cudart)
endif()

set_target_properties(${project} PROPERTIES FOLDER ${folder})