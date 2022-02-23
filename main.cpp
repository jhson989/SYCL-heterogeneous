#include <vector>
#include <iostream>
#include <algorithm>
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_device_selector.hpp>
namespace sycl = cl::sycl;

#include "include/matmul_cpu.hpp"
#include "include/matmul_gpu.hpp"
#include "include/matmul_fpga_emu.hpp"

void print_properties(sycl::queue& queue);
#define DTYPE long long
const size_t M=1024*5;
const size_t N=1024*5;
const size_t K=1024*5;



int main(void) {


    std::cout << "=================================================\n";
    std::cout << "SYCL Primitives : Parallel 2D Matrix Multiplication\n";
    std::cout << "-- 2D Matrix : A["<<M<<","<<K<<"] * B["<<K<<","<<N<<"] = C["<<M<<","<<N<<"]\n";
    std::cout << "-- total size of three 2D matrices: "<<sizeof(DTYPE)*(M*N+M*K+K*N)/1024.0/1024.0/1024.0<<" GB\n";
    std::cout << "=================================================\n\n";

    sycl::property_list properties{sycl::property::queue::enable_profiling()};

    /**********************************************************
     * Accelerator Setup
     **********************************************************/

    // CPU
    sycl::cpu_selector cpu;
    sycl::queue cpu_q(cpu, properties);
    print_properties(cpu_q);

    // GPU
    sycl::gpu_selector gpu;
    sycl::queue gpu_q(gpu, properties);
    print_properties(gpu_q);

    // FPGA emulator
    sycl::ext::intel::fpga_emulator_selector fpga;
    sycl::queue fpga_q(fpga, properties);
    print_properties(fpga_q);

    /**********************************************************
     * Data preparation
     **********************************************************/

    std::vector<DTYPE> A(M*K);
    std::generate(A.begin(), A.end(), [](){return std::rand()%10-5;});
    std::vector<DTYPE> B(K*N);
    std::generate(B.begin(), B.end(), [](){return std::rand()%10-5;});
    std::vector<DTYPE> C(M*N);

    /**********************************************************
     * Launch kernels
     **********************************************************/
    std::cout << "Elapsed time:\n";
    // CPU
    double time_cpu = cpu::matmul<DTYPE>(cpu_q, A, B, C, M, N, K);
    std::cout << "--CPU: "<<time_cpu<<" s\n";

    // GPU
    double time_gpu = gpu::matmul<DTYPE>(gpu_q, A, B, C, M, N, K);
    std::cout << "--GPU: "<<time_gpu<<" s\n";

    // FPGA emulator
    double time_fpga_emu = fpga_emu::matmul<DTYPE>(fpga_q, A, B, C, M, N, K);
    std::cout << "--FPGA emulator: "<<time_fpga_emu<<" s\n";


    return 0;
}









/****************************************************
 * Utility functions
 *****************************************************/

void print_properties(sycl::queue& queue) {

    sycl::device dev = queue.get_device();

    std::cout << "=============== Device Properties ==============" << std::endl;
    std::cout << "Name: " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Vendor: " << dev.get_info<sycl::info::device::vendor>() << std::endl;
    std::cout << "Memory size: " << dev.get_info<sycl::info::device::global_mem_size>()/1024.0f/1024.0f/1024.0f << " GB"  << std::endl;
    std::cout << "================================================" << std::endl << std::endl;
}
