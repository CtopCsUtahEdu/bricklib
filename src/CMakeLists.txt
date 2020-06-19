add_library(brick STATIC memfd.cpp)
install(TARGETS brick
        ARCHIVE DESTINATION lib)

add_library(brick-mpi STATIC brick-mpi.cpp array-mpi.cpp)
install(TARGETS brick-mpi
        ARCHIVE DESTINATION lib)

add_library(brickhelper STATIC brickcompare.cpp multiarray.cpp)
install(TARGETS brickhelper
        ARCHIVE DESTINATION lib)
