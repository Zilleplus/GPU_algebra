cmake_minimum_required(VERSION 3.20)
project(GPU_Algebra)

enable_language(CUDA)

include(${CMAKE_SOURCE_DIR}/cmake_conan/conan.cmake)

conan_cmake_configure(REQUIRES
        catch2/2.13.6
        GENERATORS
        ycm
        cmake_find_package
        cmake
        cmake_paths)

conan_cmake_autodetect(settings)

conan_cmake_install(PATH_OR_REFERENCE .
        BUILD missing
        REMOTE conancenter
        SETTINGS ${settings})

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

find_package(Catch2 REQUIRED)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory("${CMAKE_SOURCE_DIR}/demos")
add_subdirectory("${CMAKE_SOURCE_DIR}/src/gvec")
add_subdirectory("${CMAKE_SOURCE_DIR}/tests/gvec")
