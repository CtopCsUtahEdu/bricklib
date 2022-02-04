add_library(brick STATIC memfd.cpp)
install(TARGETS brick
        EXPORT brick-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME})

add_library(brick-mpi STATIC brick-mpi.cpp array-mpi.cpp)
install(TARGETS brick-mpi
        EXPORT brick-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME})

add_library(brickhelper STATIC brickcompare.cpp multiarray.cpp)
install(TARGETS brickhelper
        EXPORT brick-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME})
