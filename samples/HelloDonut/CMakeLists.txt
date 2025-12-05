set(project HelloDonut)
set(folder "samples/HelloDonut")

file(GLOB_RECURSE ${project}_src "*.cpp" "*.h")

add_executable(${project} WIN32 ${${project}_src})

target_link_libraries(${project} FluxelLib donut_app donut_engine donut_render)
set_target_properties(${project} PROPERTIES FOLDER ${folder})

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
    set_property(TARGET ${project} PROPERTY
        VS_DEBUGGER_COMMAND_ARGUMENTS "${FLUXEL_PROJECT_DIR}/assets/scenes/cornell_box/scene.gltf")
endif()
