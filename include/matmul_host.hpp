#pragma once

#include <vector>
#include <iostream>
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

namespace host {

template <typename T>
double matmul(sycl::queue queue, std::vector<T>& A, std::vector<T>& B, std::vector<T>& C, const size_t M, const size_t N, const size_t K, const size_t gsize=16) {

    /********************************************************
     *  Data initilzation
     ********************************************************/
    // A
    T* device_A = sycl::malloc_device<T>(M*K, queue);
    queue.memcpy(device_A, A.data(), M*K*sizeof(T));

    // B
    T* device_B = sycl::malloc_device<T>(K*N, queue);
    queue.memcpy(device_B, B.data(), M*K*sizeof(T));

    // C
    T* device_C = sycl::malloc_device<T>(M*N, queue);


    /********************************************************
     *  Launching kernel
     ********************************************************/
    size_t ceil_M = ((M+gsize-1)/gsize)*gsize;
    size_t ceil_N = ((N+gsize-1)/gsize)*gsize;
    auto event = queue.submit([&] (sycl::handler& cgh) {

        cgh.parallel_for(sycl::nd_range<2>({ceil_M, ceil_N}, {gsize, gsize}), [=](sycl::nd_item<2> item) {

            int m = item.get_global_id(0);
            int n = item.get_global_id(1);
            if (m<M && n<N) {
                T sum = 0;
                for (int k=0; k<K; k++) {
                    sum += device_A[m*K+k]*device_B[k*N+n];
                }
                device_C[m*N+n] = sum;
            }
        });

    });


    /********************************************************
     *  Memcpy result
     ********************************************************/
    queue.memcpy(C.data(), device_C, M*N*sizeof(T));
    queue.wait();

    auto time = 1e-9 * (event.template get_profiling_info<sycl::info::event_profiling::command_end>() -
                        event.template get_profiling_info<sycl::info::event_profiling::command_start>());
    
    return time;
}

}
