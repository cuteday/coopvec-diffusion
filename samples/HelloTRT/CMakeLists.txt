set(project HelloTRT)
set(folder "samples/HelloTRT")

file(GLOB_RECURSE ${project}_src "*.cpp" "*.h")

add_executable(${project} ${${project}_src})

target_link_libraries(${project} FluxelLib donut_app donut_engine ${TRT_RTX_LIB_NAME} ${TRT_RTX_ONNXPARSER_LIB_NAME} ${CUDA_LIBRARIES} CUDA::cudart)
set_target_properties(${project} PROPERTIES FOLDER ${folder})

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
	set_property(TARGET HelloTRT PROPERTY
    	VS_DEBUGGER_COMMAND_ARGUMENTS "--onnx=${FLUXEL_PROJECT_DIR}/assets/data/HelloTRT.onnx")
endif()

set(TRT_RTX_BIN_DIR "${TRT_RTX_ROOT}/bin")

add_custom_command(TARGET HelloTRT POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:HelloTRT>"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${TRT_RTX_BIN_DIR}" "$<TARGET_FILE_DIR:HelloTRT>"
    COMMENT "Staging TensorRT-RTX runtime DLLs")

install(TARGETS HelloTRT
    RUNTIME_DEPENDENCIES
      DIRECTORIES "${TRT_RTX_BIN_DIR}"
      PRE_INCLUDE_REGEXES "tensorrt_.*"  # optional narrowing
    RUNTIME DESTINATION bin)
