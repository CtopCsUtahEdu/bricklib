add_library(brick brick-mpi.cpp memfd.cpp array-mpi.cpp)
install(TARGETS brick
        RUNTIME DESTINATION lib)

add_library(brickhelper brickcompare.cpp multiarray.cpp)
install(TARGETS brickhelper
        RUNTIME DESTINATION lib)
