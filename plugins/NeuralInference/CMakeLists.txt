set(project NeuralInference)
set(folder "plugins/NeuralInference")

file(GLOB_RECURSE ${project}_src
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_library(${project} STATIC ${${project}_src})

add_subdirectory(Shaders)

target_include_directories(${project} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${project} PRIVATE FluxelLib donut_app donut_render donut_engine
${TRT_RTX_LIB_NAME} ${TRT_RTX_ONNXPARSER_LIB_NAME} ${CUDA_LIBRARIES} CUDA::cudart CUDA::cuda_driver CUDA::curand)
set_target_properties(${project} PROPERTIES FOLDER ${folder})
add_dependencies(${project} NeuralInferenceShaders)
target_include_directories(${project} PUBLIC ${TRT_RTX_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
	set_property(TARGET NeuralInference PROPERTY
    	VS_DEBUGGER_COMMAND_ARGUMENTS "")
endif()

