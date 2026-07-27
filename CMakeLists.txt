cmake_minimum_required(VERSION 3.10)
project(Supercell_Mean_Field_Calculation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Find required packages
find_package(MPI REQUIRED)
find_package(GSL REQUIRED)

# Intel MKL configuration
# Option 1: If using Intel compiler
find_package(MKL)

# If MKL not found, try to link manually
if(NOT MKL_FOUND)
    # Set MKL root directory (adjust if needed)
    if(DEFINED MKLROOT)
        set(MKL_INCLUDE_DIRS ${MKLROOT}/include)
        set(MKL_LIBRARIES 
            ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a
            ${MKLROOT}/lib/intel64/libmkl_core.a
            ${MKLROOT}/lib/intel64/libmkl_sequential.a
            m
            dl
        )
    else()
        message(WARNING "MKL not found. Set MKLROOT environment variable or install Intel MKL")
        # Fallback to system BLAS/LAPACK if available
        find_package(BLAS)
        find_package(LAPACK)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MPI_CXX_INCLUDE_PATH}
    ${GSL_INCLUDE_DIRS}
)

if(DEFINED MKL_INCLUDE_DIRS)
    include_directories(${MKL_INCLUDE_DIRS})
endif()

# Source files
set(COMMON_SOURCES
    src/bcs_k_randomHopHF_mpi.cpp
    src/bcs_k_randomOnlyHopHF_mpi.cpp
    src/bcs_k_randomUniPotHF_mpi.cpp
    src/SFW_cal_k_randomHop_mpi.cpp
    src/SFW_cal_k_randomOnlyHop_mpi.cpp
    src/SFW_cal_k_randomUniPot_mpi.cpp
)

# Determine MKL or LAPACK libraries
if(MKL_FOUND)
    set(LAPACK_LIBRARIES ${MKL_LIBRARIES})
elseif(DEFINED MKL_LIBRARIES)
    set(LAPACK_LIBRARIES ${MKL_LIBRARIES})
elseif(LAPACK_FOUND)
    set(LAPACK_LIBRARIES ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
else()
    message(FATAL_ERROR "Neither MKL nor LAPACK found!")
endif()

# Create executables for each source file
foreach(source ${COMMON_SOURCES})
    get_filename_component(target_name ${source} NAME_WE)
    add_executable(${target_name} ${source})
    
    target_link_libraries(${target_name}
        ${MPI_CXX_LIBRARIES}
        ${GSL_LIBRARIES}
        ${LAPACK_LIBRARIES}
        m
    )
    
    target_compile_options(${target_name} PRIVATE ${MPI_CXX_COMPILE_FLAGS})
endforeach()

# Compiler-specific optimizations for MPI and Math libraries
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    add_compile_options(-mkl)
endif()
