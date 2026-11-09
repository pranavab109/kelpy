cmake_minimum_required(VERSION 3.22)
project(kelpy LANGUAGES CXX)

# 1. Force C++26 standard optimization flags
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Pull in pybind11 configuration
find_package(pybind11 CONFIG REQUIRED)

# 3. Native Cross-Platform OpenMP Detection
find_package(OpenMP REQUIRED)

# 4. Define the Python extension module
pybind11_add_module(kelpy kelpy.cpp)

# 5. Apply architecture-specific performance optimizations
if(MSVC)
    target_compile_options(kelpy PRIVATE /O2 /ffast-math /Wall)
else()
    target_compile_options(kelpy PRIVATE -O3 -march=native -ffast-math -Wall)
endif()

# 6. Bind OpenMP headers and libraries to your extension target
target_link_libraries(kelpy PRIVATE OpenMP::OpenMP_CXX)

