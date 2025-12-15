set(project BasicRenderer)
set(folder "plugins/BasicRenderer")

file(GLOB_RECURSE ${project}_src
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_library(${project} STATIC ${${project}_src})

target_include_directories(${project} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${project} PRIVATE FluxelLib donut_app donut_render donut_engine)
set_target_properties(${project} PROPERTIES FOLDER ${folder})

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
	set_property(TARGET BasicRenderer PROPERTY
    	VS_DEBUGGER_COMMAND_ARGUMENTS "")
endif()

