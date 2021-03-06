# CMake bianry version
cmake_minimum_required(VERSION 3.10)
# Proejct info
project(
    SYCL_CPU_GPU_HYBRID_COMPUTING
    VERSION 0.1
    DESCRIPTION "SYCL CPU-GPU hybrid example"
)

set(PATH_SYCL_BUILD /data/share/oneapi/llvm/build)
set(CMAKE_CXX_COMPILER ${PATH_SYCL_BUILD}/bin/clang++)
set(PATH_SYCL_INC ${PATH_SYCL_BUILD}/include/sycl)
set(PATH_SYCL_LIB ${PATH_SYCL_BUILD}/lib)
set(SYCL_COMPILE_OPTION -fsycl -fsycl-targets=spir64,nvptx64-nvidia-cuda,spir64_fpga)

set(APP hybrid.out)
set(MAIN main.cpp)

add_executable(${APP} ${MAIN} ${SRC_FILES})

target_include_directories(${APP} PUBLIC ${PATH_SYCL_INC} ../../include/)
target_compile_options(${APP} PUBLIC ${SYCL_COMPILE_OPTION})

target_link_libraries(${APP} PUBLIC sycl)
target_link_directories(${APP} PUBLIC ${PATH_SYCL_LIB})
target_link_options(${APP} PUBLIC ${SYCL_COMPILE_OPTION})