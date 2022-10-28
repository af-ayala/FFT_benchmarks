
execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE rocfft_cxx_version)
string(REGEX MATCH "^HIP" rocfft_haship "${rocfft_cxx_version}")

if (NOT rocfft_haship)
    message(WARNING "rocfft_ENABLE_ROCM requires that the CMAKE_CXX_COMPILER is set to the Rocm hipcc compiler.")
endif()

get_filename_component(rocfft_hipccroot ${CMAKE_CXX_COMPILER} DIRECTORY)
get_filename_component(rocfft_hipccroot ${rocfft_hipccroot} DIRECTORY)

set(rocfft_ROCM_ROOT "${rocfft_hipccroot}" CACHE PATH "The root folder for the Rocm framework installation")
list(APPEND CMAKE_PREFIX_PATH "${rocfft_ROCM_ROOT}")

find_package(rocfft REQUIRED)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RocFFT DEFAULT_MSG ROCFFT_LIBRARIES)
