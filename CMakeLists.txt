
cmake_minimum_required(VERSION 3.22)
project(kelpy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Modern Python discovery
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find Homebrew OpenMP on macOS
if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE OPENMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE BREW_RESULT
    )

    if(BREW_RESULT EQUAL 0)
        list(PREPEND CMAKE_PREFIX_PATH "${OPENMP_PREFIX}")
        set(OpenMP_ROOT "${OPENMP_PREFIX}")
    endif()
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

# Build Python extension
pybind11_add_module(kelpy src/kelpy/kelpy.cpp)

# Compiler optimizations
if(MSVC)
    target_compile_options(kelpy PRIVATE /O2 /fp:fast)
else()
    target_compile_options(kelpy PRIVATE -O3 -ffast-math -Wall)
endif()

target_link_libraries(kelpy PRIVATE OpenMP::OpenMP_CXX)

# Install extension into wheel
install(TARGETS kelpy
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
