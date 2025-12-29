set(LIBRARY_FILTER "src")

set(project "FluxelLib")
set(folder "${LIBRARY_FILTER}")

# Collect sources per folder so they can be toggled independently
set(SRC_FILES "")

file(GLOB_RECURSE CORE_SRC "Core/*.cpp" "Core/*.h")
list(APPEND SRC_FILES ${CORE_SRC})

file(GLOB_RECURSE RENDER_SRC "Render/*.cpp" "Render/*.h")
list(APPEND SRC_FILES ${RENDER_SRC})

file(GLOB_RECURSE UTILS_SRC "Utils/*.cpp" "Utils/*.h")
list(APPEND SRC_FILES ${UTILS_SRC})

add_subdirectory(Core/Math)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Core/Config.h.in ${CMAKE_CURRENT_BINARY_DIR}/Include/Config.h)

add_library(${project} STATIC EXCLUDE_FROM_ALL ${SRC_FILES})
target_include_directories(${project} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Core
    ${CMAKE_CURRENT_BINARY_DIR}/Include
)

target_link_libraries(${project} PUBLIC donut_app donut_engine krr_math)

set_target_properties(${project} PROPERTIES FOLDER ${folder})
