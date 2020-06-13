#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cuda_runtime.h>// CUDA utilities and system includes
#include <helper_functions.h>  // CUDA SDK Helper functions
#include <helper_cuda.h>       // CUDA device initialization helper functions
#include <helper_image.h>
#include <cuda_profiler_api.h>

#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
//#include "Timer.h"

using namespace cv;
using namespace std;




//Kernel for the color conversion RGB to YCbCr and to compute the histogram of the Y channel.
inline __global__ void RGB_to_YCbCr_kernel(unsigned char* d_ptr_image, int* d_hist, int width, int height) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long pixel_idx;

	//number of threads in a block * number of blocks
	int step = blockDim.x * gridDim.x;

	//If doesn't have the required number of threads, the access to the image in global memory is coalesced.
	//The image is saved in a buffer in order to ease the coalesced access;
	for (int i = idx; i < width * height; i += step) {

		//to found the pixel position multiply for the number of channels (3)
		pixel_idx = i * 3;
		int r = d_ptr_image[pixel_idx + 0];
		int g = d_ptr_image[pixel_idx + 1];
		int b = d_ptr_image[pixel_idx + 2];

		int Y = (int)(0.299 * r + 0.587 * g + 0.114 * b);
		int Cb = (int)(128 - 0.168736 * r - 0.331264 * g + 0.5 * b);
		int Cr = (int)(128 + 0.5 * r - 0.418688 * g - 0.081312 * b);

		d_ptr_image[pixel_idx + 0] = Y;
		d_ptr_image[pixel_idx + 1] = Cb;
		d_ptr_image[pixel_idx + 2] = Cr;

		atomicAdd(&d_hist[Y], 1);
	}
	__syncthreads();
}



//This kernel equalizes the histogram
inline __global__ void equalize_kernel(int* d_cdf, int* d_hist_eq, int width, int height) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = idx; i < 256; i += blockDim.x * gridDim.x) {
		//formula for calculate the equalized histogram with the cumulative distribution function
		d_hist_eq[i] = (int)((((float)d_cdf[i] - d_cdf[0]) / (((float)width * height - 1))) * 255);
	}
}



//This kernel maps the new equalized values of the Y channel and
// makes the color conversion from YCbCr to RGB.
inline __global__ void YCbCr_to_RGB_kernel(unsigned char* d_ptr_image, int* d_hist_eq, int width, int height) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long pixel_idx;

	int step = blockDim.x * gridDim.x;

	for (int i = idx; i < width * height; i += step) {

		pixel_idx = i * 3;

		//of y make the new value, of cb and cr make the old value		
		int y = d_hist_eq[d_ptr_image[pixel_idx] + 0];
		int cb = d_ptr_image[pixel_idx + 1];
		int cr = d_ptr_image[pixel_idx + 2];

		//use of new equalized y value for recalculate the rgb channels
		int R = max(0, min(255, (int)(y + 1.402 * (cr - 128))));
		int G = max(0, min(255, (int)(y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128))));
		int B = max(0, min(255, (int)(y + 1.772 * (cb - 128))));

		d_ptr_image[pixel_idx + 0] = R;
		d_ptr_image[pixel_idx + 1] = G;
		d_ptr_image[pixel_idx + 2] = B;

	}
}


/*
//Check the return value of the CUDA runtime API call and exit the application if the call has failed.

static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
	exit(1);
}
*/


inline static int eq_GPU(unsigned char* ptr_image, int width, int height) {

	unsigned char* d_ptr_image;		//immagine
	int* d_hist;					//istogramma
	int* d_cdf;						//funzione distribuzione cumulativa
	int* d_hist_eq;					//istogramma equalizzato
	int hist[256] = { 0 };

	//Allocate the GPU global memory needed.
	cudaMalloc((void**)&d_ptr_image, sizeof(char) * (width * height * 3));
	cudaMalloc((void**)&d_hist, sizeof(int) * (256));
	cudaMalloc((void**)&d_hist_eq, sizeof(int) * (256));
	cudaMalloc((void**)&d_cdf, sizeof(int) * (256));

	//Copy the image buffer to the global memory.
	cudaMemcpy(d_ptr_image, ptr_image, sizeof(char) * (width * height * 3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_hist, hist, sizeof(int) * (256), cudaMemcpyHostToDevice);

	int n_threads = 256;
	//calculate of the number of blocks in relation of number of threads
	//int in way to have a defect approximation
	int n_blocks = (width * height + (n_threads - 1)) / n_threads;

	RGB_to_YCbCr_kernel << < n_blocks, n_threads >> > (d_ptr_image, d_hist, width, height);

	//Copy to host the histogram computed in the first kernel.
	cudaMemcpy(hist, d_hist, sizeof(int) * (256), cudaMemcpyDeviceToHost);

	//calculate of cdf function
	int sum = 0;
	int cdf[256] = { 0 };

	for (int i = 0; i < 256; i++) {
		sum += hist[i];
		cdf[i] = sum;
	}

	cudaMemcpy(d_cdf, cdf, sizeof(int) * (256), cudaMemcpyHostToDevice);

	equalize_kernel << < n_blocks, n_threads >> > (d_cdf, d_hist_eq, width, height);
	YCbCr_to_RGB_kernel << < n_blocks, n_threads >> > (d_ptr_image, d_hist_eq, width, height);

	//Copy to host the equalized image.
	cudaMemcpy(ptr_image, d_ptr_image, sizeof(char) * (width * height * 3), cudaMemcpyDeviceToHost);

	//Release GPU memory.
	cudaFree(d_ptr_image);
	cudaFree(d_hist);
	cudaFree(d_hist_eq);
	cudaFree(d_cdf);

	return 0;
}






