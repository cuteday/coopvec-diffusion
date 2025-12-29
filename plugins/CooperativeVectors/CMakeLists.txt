set(project CooperativeVectors)
set(folder "plugins/CooperativeVectors")

file(GLOB_RECURSE ${project}_src
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_library(${project} STATIC ${${project}_src})

add_subdirectory(Shaders)

add_dependencies(${project} CooperativeVectorsShaders)
target_include_directories(${project} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${project} PRIVATE FluxelLib donut_app donut_render donut_engine)
set_target_properties(${project} PROPERTIES FOLDER ${folder})

if (DONUT_WITH_DX12)
    add_dependencies(${project} dx12-agility-sdk)
    target_compile_definitions(
		${project} PRIVATE DONUT_D3D_AGILITY_SDK_VERSION=${DONUT_D3D_AGILITY_SDK_VERSION}
		${project} PRIVATE DONUT_D3D_AGILITY_PREVIEW_SDK_VERSION=${DONUT_D3D_AGILITY_PREVIEW_SDK_VERSION}
	)
endif()

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
	set_property(TARGET CooperativeVectors PROPERTY
    	VS_DEBUGGER_COMMAND_ARGUMENTS "")
endif()

