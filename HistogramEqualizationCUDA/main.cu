#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cuda_runtime.h>
#include <helper_functions.h>  
#include <helper_cuda.h>       
#include <helper_image.h>
#include <cuda_profiler_api.h>

#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>

#include "HistogramEqualizationGPU.cu"


using namespace cv;
using namespace std;


struct GpuTimer {
	cudaEvent_t start;
	cudaEvent_t stop;
	GpuTimer() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}
	~GpuTimer() {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	void Start() {
		cudaEventRecord(start, 0);
	}
	void Stop() {
		cudaEventRecord(stop, 0);
	}
	float Elapsed() {
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};




int main(void)
{
	Mat image = imread("images/image1.jpg");


	namedWindow("Original Image", 0);
	resizeWindow("Equalized Image", 800, 600);
	imshow("Original Image", image);

	int height = image.rows;
	int width = image.cols;

	//Convert the image into a buffer
	unsigned char* ptr_image = image.ptr();

	//Start GPU timer
	GpuTimer timer;
	timer.Start();

	cout << "Processing with GPU..." << endl;
	eq_GPU(ptr_image, width, height);

	timer.Stop();
	cout << "Time for GPU: " << timer.Elapsed() << " msec" << endl << endl;

	cout << "Saving equalized image..." << endl;
	string equalized;
	equalized = "images/image_equalized_cuda.jpg";
	imwrite(equalized, image);
	cout << "Image saved" << endl;


	namedWindow("Equalized Image", 0);
	resizeWindow("Equalized Image", 800, 600);
	imshow("Equalized Image", image);

	waitKey();

	return 0;
}