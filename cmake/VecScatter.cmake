# Configure Vector Scatter

find_package(Python 3.6 REQUIRED COMPONENTS Interpreter)
set(VecScatter_SCRIPT ${PROJECT_SOURCE_DIR}/codegen/vecscatter)
set(VecScatter_MODULE ${PROJECT_SOURCE_DIR}/codegen)

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
    add_custom_command(OUTPUT ${Output}
            COMMAND ${CMAKE_COMMAND} -E env ${Python_EXECUTABLE} ${VecScatter_SCRIPT} ${Input} ${Output} -- ${CMAKE_CXX_FLAGS_LIST} ${${Name}_COMPILE_OPTIONS} ${VSTARGET_INC}
            VERBATIM
            MAIN_DEPENDENCY ${Input}
            COMMENT "[VS][${Name}] Vector Scatter transformation"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set(VSTARGET_${Name}_OUTPUT ${Output})
endmacro()
