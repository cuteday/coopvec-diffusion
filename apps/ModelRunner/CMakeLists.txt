set(project ModelRunner)
set(folder "apps/ModelRunner")

file(GLOB_RECURSE ${project}_src
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_executable(${project} ${${project}_src})

target_link_libraries(${project} FluxelLib BasicRenderer NeuralInference donut_app donut_render donut_engine ${TRT_RTX_LIB_NAME} ${TRT_RTX_ONNXPARSER_LIB_NAME} ${CUDA_LIBRARIES} CUDA::cudart CUDA::cuda_driver)
set_target_properties(${project} PROPERTIES FOLDER ${folder})

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
	set_property(TARGET ModelRunner PROPERTY
    	VS_DEBUGGER_COMMAND_ARGUMENTS "")
endif()

set(TRT_RTX_BIN_DIR "${TRT_RTX_ROOT}/bin")

add_custom_command(TARGET ModelRunner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:ModelRunner>"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${TRT_RTX_BIN_DIR}" "$<TARGET_FILE_DIR:ModelRunner>"
    COMMENT "Staging TensorRT-RTX runtime DLLs")

install(TARGETS ModelRunner
    RUNTIME_DEPENDENCIES
      DIRECTORIES "${TRT_RTX_BIN_DIR}"
      PRE_INCLUDE_REGEXES "tensorrt_.*"  # optional narrowing
    RUNTIME DESTINATION bin)
