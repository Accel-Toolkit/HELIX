cmake_minimum_required(VERSION 3.15)
project(linac_gen_csrc)

find_package(pybind11 REQUIRED)
find_package(OpenMP)

pybind11_add_module(_pic_kernels pic_kernels.cpp)
target_compile_options(_pic_kernels PRIVATE -O3 -march=native)

if(OpenMP_CXX_FOUND)
    message(STATUS "Building _pic_kernels with OpenMP (${OpenMP_CXX_VERSION})")
    target_link_libraries(_pic_kernels PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found — _pic_kernels will be single-threaded")
endif()

# Field-map sampling kernel.  -ffp-contract=off is REQUIRED: the kernel's
# bit-identity contract with scipy RGI (and with the pure-python fallback)
# forbids the compiler from fusing f*w + acc into FMA instructions.
pybind11_add_module(_fieldmap_kernels fieldmap_kernels.cpp)
target_compile_options(_fieldmap_kernels PRIVATE -O3 -march=native -ffp-contract=off)

if(OpenMP_CXX_FOUND)
    message(STATUS "Building _fieldmap_kernels with OpenMP")
    target_link_libraries(_fieldmap_kernels PRIVATE OpenMP::OpenMP_CXX)
endif()
