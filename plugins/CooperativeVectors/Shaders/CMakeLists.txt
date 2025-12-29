set(project CooperativeVectorsShaders)
set(folder "plugins/CooperativeVectors/Shaders")

file(GLOB shaders "*.slang")

set_source_files_properties(${shaders} PROPERTIES VS_TOOL_OVERRIDE "None")
add_custom_target(${project}
	DEPENDS ShaderMake
    SOURCES ${shaders})
	set_target_properties(${project} PROPERTIES
	FOLDER ${folder}
)

set(FLUXEL_COOPVEC_SHADER_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE PATH "" FORCE)
