# Configure Vector Scatter

find_package(Python 3.6 REQUIRED COMPONENTS Interpreter)
set(VecScatter_SCRIPT ${BRICK_PROJECT_PATH}/codegen/vecscatter)
set(VecScatter_MODULE ${BRICK_PROJECT_PATH}/codegen)
set(ENV{VecScatter_MODULE} ${BRICK_PROJECT_PATH}/codegen)

set(VS_PREPROCESSOR cpp CACHE STRING "Preprocessor for vector scatter")

macro(VSTARGET Name Input Output)
    get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
    set(VSTARGET_INC "")
    foreach (dir ${dirs})
        list(APPEND VSTARGET_INC -I${dir})
    endforeach ()
    foreach (dir ${${Name}_INCLUDE_DIRS})
        list(APPEND VSTARGET_INC -I${dir})
    endforeach ()
    string(REPLACE " " ";" CMAKE_CXX_FLAGS_LIST ${CMAKE_CXX_FLAGS})
    if (IS_ABSOLUTE "${Output}")
        set(VSTARGET_${Name}_OUTPUT "${Output}")
    else()
        set(VSTARGET_${Name}_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/${Output}")
    endif()
    add_custom_command(OUTPUT "${VSTARGET_${Name}_OUTPUT}"
            COMMAND ${CMAKE_COMMAND} -E env VSCPP=${VS_PREPROCESSOR} ${Python_EXECUTABLE} ${VecScatter_SCRIPT} "${Input}" "${VSTARGET_${Name}_OUTPUT}" -- ${CMAKE_CXX_FLAGS_LIST} ${${Name}_COMPILE_OPTIONS} ${VSTARGET_INC}
            VERBATIM
            MAIN_DEPENDENCY "${Input}"
            COMMENT "[VS][${Name}] Vector Scatter transformation"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endmacro()
