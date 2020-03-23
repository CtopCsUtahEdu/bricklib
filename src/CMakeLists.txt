add_library(brick STATIC brick-mpi.cpp memfd.cpp array-mpi.cpp)
install(TARGETS brick
        ARCHIVE DESTINATION lib)

add_library(brickhelper STATIC brickcompare.cpp multiarray.cpp)
install(TARGETS brickhelper
        ARCHIVE DESTINATION lib)
