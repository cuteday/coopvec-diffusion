set(LIBRARY_FILTER "src")

# Recursively glob all .cpp files in the src directory
file(GLOB_RECURSE SRC_FILES "*.cpp" "*.h")

set(project "FluxelLib")
set(folder "${LIBRARY_FILTER}")

add_library(${project} STATIC EXCLUDE_FROM_ALL ${SRC_FILES})
target_include_directories(${project} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${project} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_link_libraries(${project} PUBLIC donut_app donut_engine)

set_target_properties(${project} PROPERTIES FOLDER ${folder})