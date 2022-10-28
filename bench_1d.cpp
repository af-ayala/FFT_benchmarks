/*
cmake -DENABLE_CUFFT=ON .; make -j; ./bench_1d
cmake -DENABLE_CUFFT = ON ..
*/

#include <iostream>
#include <string>
#include <algorithm>
#include <iterator>
#include <array>
#include <vector>
#include <complex>
#include <random>
#include <chrono>

#if defined(ENABLE_CUFFT)
    #include <cufft.h>
    #include <cuda_runtime_api.h>
    #define data_type   cufftDoubleComplex
    #define plan_type   cufftHandle
    #define host_malloc(h_, size_) cudaMallocHost((void**) &(h_), (size_));
    #define device_malloc(g_, size_) cudaMalloc((void**) &(g_), (size_));
    #define host_mfree(h_) cudaFreeHost((h_));
    #define device_mfree(g_) cudaFree((g_));
    #define gpu_copy_h2d(h_, g_, size_) cudaMemcpy((g_), (h_), (size_), cudaMemcpyHostToDevice)
    #define gpu_copy_d2h(h_, g_, size_) cudaMemcpy((h_), (g_), (size_), cudaMemcpyDeviceToHost)
#elif defined(ENABLE_ROCFFT)
    #ifndef __HIP_PLATFORM_HCC__
    #define __HIP_PLATFORM_HCC__
    #endif
    #include <hip/hip_runtime.h>
    #include <rocfft.h>
    #define data_type   double
    #define plan_type   double
    #define host_malloc(h_, size_)
    #define device_malloc(g_, size_)
    #define host_mfree(h_)
    #define device_mfree(g_)
    #define gpu_copy_h2d(h_, g_, size_)
    #define gpu_copy_d2h(h_, g_, size_)
#elif defined(ENABLE_VKFFT)
    #define data_type   double
    #define plan_type   double
    #define host_malloc(h_, size_)
    #define device_malloc(g_, size_)
    #define host_mfree(h_)
    #define device_mfree(g_)
    #define gpu_copy_h2d(h_, g_, size_)
    #define gpu_copy_d2h(h_, g_, size_)
#else
    #define data_type   double
    #define plan_type   double
    #define host_malloc(h_, size_)
    #define device_malloc(g_, size_)
    #define host_mfree(h_)
    #define device_mfree(g_)
    #define gpu_copy_h2d(h_, g_, size_)
    #define gpu_copy_d2h(h_, g_, size_)
#endif


int main()
{
    std::vector<int> N {256, 256, 256};
    int niter = 5;

    // size_t fft_size = N[0]*N[1]*N[2];
    size_t fft_size = 6;
    std::vector<std::complex<double>> input(fft_size);
    std::vector<std::complex<double>> output(fft_size);

    std::vector<double> time_cufft(niter,0.0), time_rocfft(niter,0.0), time_vkfft(niter,0.0), time_fftw(niter,0.0);

    data_type *d_data_in = NULL, *d_data_out = NULL;
    plan_type plan;

    size_t mem_size = sizeof(data_type) * fft_size;
    device_malloc(d_data_in,  mem_size);
    device_malloc(d_data_out, mem_size);

    std::minstd_rand park_miller(1234);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    for(auto &e : input)
        e = static_cast<double>(unif(park_miller));

    for(int i=0; i < input.size(); i++){
        std::cout << input.at(i) << ' ';
        // std::cout << input.at(i).real() << "+ 1i" << input.at(i).imag() <<  ' ';
    }
    std::cout<< "\n";

    gpu_copy_h2d(input.data(), d_data_in, mem_size);

    #if defined(ENABLE_CUFFT)
        // cufftPlan3d(&plan, N[0], N[1], N[2], CUFFT_Z2Z);
        cufftPlan1d(&plan, fft_size, CUFFT_Z2Z, 1);

        for (size_t i = 0; i < niter; i++){
            auto start = std::chrono::steady_clock::now();
            cudaDeviceSynchronize();
            cufftExecZ2Z(plan, d_data_in, d_data_out, CUFFT_FORWARD);
            cudaDeviceSynchronize();
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            time_cufft.at(i)= elapsed_seconds.count();
        }

    #elif defined(ENABLE_ROCFFT)
                
    #endif

    gpu_copy_d2h(output.data(), d_data_out, mem_size);

    for(int i=0; i < output.size(); i++)
        std::cout << output.at(i) << ' ';
    std::cout<< "\n";

    // Inverse FFT
    #if defined(ENABLE_CUFFT)
    cufftExecZ2Z(plan, d_data_out, d_data_out, CUFFT_INVERSE);
    #endif

    // Check Error  \| X - IFFT(X) \|
    gpu_copy_d2h(output.data(), d_data_out, mem_size);

    for(int i=0; i < output.size(); i++)
        std::cout << output.at(i) << ' ';
    std::cout<< "\n";
    
    double norm_i, e = 0.0;

    for(int i=0; i < output.size(); i++){
        norm_i = std::sqrt( std::pow(input.at(i).real() - output.at(i).real()/fft_size , 2) + std::pow(input.at(i).imag() - output.at(i).imag()/fft_size, 2) );
        e = std::max(e, norm_i);
    }
    
    std::cout<< "\n==========================================================\n";
    std::cout<< "\t\t\tRuntime (s)\n";
    std::cout<<"cuFFT\t\trocFFT\t\tvkFFT\t\tFFTW\n";
    std::cout<< "==========================================================\n";

    for(int i=0; i < niter; i++)
        std::cout << time_cufft[i] << "\t"  << time_rocfft[i] << "\t" << time_vkfft[i] << "\t" << time_fftw[i] << std::endl;

    std::cout<< "\nSize: ";
    for(int i=0; i < N.size(); i++)
        std::cout << N.at(i) << 'x';

    std::cout<< "\t |FFT(X)-iFFT(X)|_{infty}: " << e << std::endl;

}