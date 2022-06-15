add_library(gvec STATIC
        vec.h
        vec.cu
        )

target_include_directories(gvec PUBLIC "${CMAKE_SOURCE_DIR}/src")
