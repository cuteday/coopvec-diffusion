set(project HelloWorld)
set(folder "samples/HelloWorld")

file(GLOB_RECURSE ${project}_src "*.cpp" "*.h")

add_executable(${project} ${${project}_src})

target_link_libraries(${project} FluxelLib donut_app donut_engine)
set_target_properties(${project} PROPERTIES FOLDER ${folder})

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
endif()