set(LIBRARY_FILTER "src")

# Recursively glob all .cpp files in the src directory
file(GLOB_RECURSE SRC_FILES "*.cpp" "*.h")

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

set_target_properties(${project} PROPERTIES FOLDER ${folder})