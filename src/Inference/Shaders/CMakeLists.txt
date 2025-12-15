set(project NeuralInferenceShaders)
set(folder "${LIBRARY_FILTER}/Inference/Shaders")

file(GLOB shaders "*.slang")

set_source_files_properties(${shaders} PROPERTIES VS_TOOL_OVERRIDE "None")
add_custom_target(${project}
	DEPENDS ShaderMake
    SOURCES ${shaders})
	set_target_properties(${project} PROPERTIES
	FOLDER ${folder}
)

set(FLUXEL_NEURAL_INFERENCE_SHADER_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE PATH "" FORCE)

set(shader_includes
	${FLUXEL_MAIN_SHADER_INCLUDE_DIR}
	${FLUXEL_NEURAL_INFERENCE_SHADER_INCLUDE_DIR}
	${CMAKE_CURRENT_LIST_DIR}
)

set(SHADER_COMPILE_OPTIONS "--matrixRowMajor --hlsl2021" )

set(SHADER_COMPILE_OPTIONS_SPIRV " -X  \"-Wno-41017\" " )

set(SHADER_COMPILE_OPTIONS_DXIL " --shaderModel 6_8 --hlsl2021 -X \"-Wno-41012 -Wno-41016 -Wno-41017 -Xdxc -Vd\" " )

donut_compile_shaders_all_platforms(
    TARGET ${project}
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
	INCLUDES ${shader_includes}
    FOLDER ${folder}
    OUTPUT_BASE ${FLUXEL_BINARY_DIR}/shaders/${project}
	SHADERMAKE_OPTIONS ${SHADER_COMPILE_OPTIONS}
    SHADERMAKE_OPTIONS_SPIRV ${SHADER_COMPILE_OPTIONS_SPIRV}
	SHADERMAKE_OPTIONS_DXIL ${SHADER_COMPILE_OPTIONS_DXIL}
    SOURCES ${shaders}
    SLANG
)
