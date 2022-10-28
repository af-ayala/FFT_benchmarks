cmake_minimum_required(VERSION 3.10)

project("FFT_Benchmarks" VERSION 0.1 LANGUAGES CXX)

#################
# Default Options
#################

option(ENABLE_ROCFFT       "Enable the ROCFFT backend"       OFF)
option(ENABLE_CUFFT        "Enable the CUFFFT backend"       OFF)
option(ENABLE_FFTW         "Enable the FFTW backend"         OFF)

set(CMAKE_C_FLAGS "-lm")

#####################
# Library definitions
#####################

if (ENABLE_ROCFFT)
    add_definitions(-DENABLE_ROCFFT)
    set(ROCM_FLAGS "" CACHE STRING "added flags specific to the rocFFT backend")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ROCM_FLAGS}")
endif()

if (ENABLE_CUFFT)
    add_definitions(-DENABLE_CUFFT)
    find_package(CUDA REQUIRED)
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
endif()

if (ENABLE_FFTW)
    add_definitions(-DENABLE_FFTW)
endif()


macro(add_benchmark benchmark)

    add_executable(${benchmark} "${benchmark}.cpp")

    target_include_directories(${benchmark} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>)
    
    if (ENABLE_CUFFT)
        cuda_add_cufft_to_target(${benchmark})
        target_include_directories(${benchmark} PUBLIC ${CUDA_INCLUDE_DIRS})
        target_link_libraries(${benchmark} ${CUDA_LIBRARIES})
    endif()

    if (ENABLE_ROCFFT)
        target_link_libraries(${benchmark} roc::rocfft)

    endif()
    if (ENABLE_FFTW)
        target_link_libraries(${benchmark} fftw3)
    endif()

endmacro()


##################
# Build benchmarks
##################
add_benchmark(bench_1d)


###############
# Documentation
###############
if (ENABLE_DOXYGEN)
    add_subdirectory(doxygen)
endif()

# Print summary of the CMake options
if (${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/print_summary.cmake)
endif()
