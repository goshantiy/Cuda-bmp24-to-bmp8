#include "bmp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
void printDeviceProp()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);//определение	параметров GPU с номером 0
	printf("Device name : %s\n", deviceProp.name);
	printf("Total global memory : %d MB\n",
		deviceProp.totalGlobalMem / 1024 / 1024);
	printf("Shared memory per block : %d\n",
		deviceProp.sharedMemPerBlock);
	printf("Registers per block : %d\n",
		deviceProp.regsPerBlock);
	printf("Warp size : %d\n", deviceProp.warpSize);
	printf("Memory pitch : %d\n", deviceProp.memPitch);
	printf("Max threads per block : %d\n",
		deviceProp.maxThreadsPerBlock);
	printf("Max threads dimensions : x = %d, y = %d, z =% d\n", deviceProp.maxThreadsDim[0],
		deviceProp.maxThreadsDim[1],
		deviceProp.maxThreadsDim[2]);
	printf("Max grid size: x = %d, y = %d, z = %d\n",
		deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
		deviceProp.maxGridSize[2]);
	printf("Clock rate: %d\n", deviceProp.clockRate);
	printf("Total constant memory: %d\n",
		deviceProp.totalConstMem);
	printf("Compute capability: %d.%d\n",
		deviceProp.major, deviceProp.minor);
	printf("Texture alignment: %d\n",
		deviceProp.textureAlignment);
	printf("Device overlap: %d\n",
		deviceProp.deviceOverlap);
	printf("Multiprocessor count: %d\n",
		deviceProp.multiProcessorCount);
	printf("Kernel execution timeout enabled: %s\n", deviceProp.kernelExecTimeoutEnabled ? "true" :
		"false");
	scanf("");
}


__constant__ int color_palette[256];




/*
INPUT: COLORS CONVERTED TO INT FROM GPU, POINTER TO RESULTING PALETTE, SIZE OF COLORS ARRAY, NUM OF COLORS THAT EACH THREAD PROCEED
OUTPUT: COLOR PALETTE
the task boils down to finding unique numbers in an unsorted array
each value from colors array inserted to hash table by index: int COLOR * 0xDEADBEEF >> 19 
if value in hash table by index is busy, thread tries to find new index: index = (index + 1) & 4095;
if this hash table's value busy too, thread tries next indexes
*/

__global__ void d_createColorPalette(int* d_all_colors, int* d_color_palette, size_t size,int num)
{
	__shared__ int temp_colors[10000];
	int tid = threadIdx.x;
	int block = tid*num;
	if (block < size)
	{
		/*for (int i = 0; i < num && block + i < 10000; i++)
			temp_colors[block + i] = -1;*/
		for (int i = 0; i < num && block + i < size; i++)
		{
			if (d_all_colors[block + i] != d_all_colors[block + i - 1])
			{
				int index = (d_all_colors[block + i] * 0xDEADBEEF) >> 19;
				if (temp_colors[index] == 0)
				{
					temp_colors[index]= d_all_colors[block + i];
				}
				else
				{
					if (temp_colors[index] != d_all_colors[block + i])
					{
						index = (index + 1) & 4095;
						while (temp_colors[index] != 0 && index < 10000)
							index++;
						if (index == 10000)
						{
							index = 0;
							while (temp_colors[index] != 0 && index < 10000)
								index++;
						}
						temp_colors[index] = d_all_colors[block + i];
					}
				}
			}
		}
	}
	__syncthreads();	
	if (threadIdx.x == 0)
	{
		int color_palette[256];
		UINT8 ix = 1;
		color_palette[0] = temp_colors[0];
		for (int i = 1; i < 10000; i++)
		{
			//if (temp_colors[i] != -1)
			{
				bool check = true;
				for (int j = 0; (j < ix) && check; ++j)
				{
					if (color_palette[j] == temp_colors[i])
						check = false;
				}
				if (check)
				{
					color_palette[ix] = temp_colors[i];
					ix++;
				}
			}
		}
		for (int i = 0; i < 256; i++)
			d_color_palette[i] = color_palette[i];
	}
}
__global__ void d_applyPalette(int* d_all_colors, int* d_color_palette, UINT8* d_result, size_t size)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	if (ix < size)
		for (int i = 0; i < 256; i++)
			if (d_all_colors[ix] == d_color_palette[i])
			{
				d_result[ix] = i;
				break;
			}
}
//CLEANS GPU PALETTE FROM ZEROES
void cleanPalette(std::vector<int> &colors)
{
	auto vec = std::unique(colors.begin(), colors.end());
	colors.erase(vec, colors.end());
	if (colors.end() == colors.begin());
	colors.resize(colors.size() - 1);
}
void printIntPaletteToRgb(std::vector<int>& colors)
{
	int count = 0;
	for (auto v : colors)
	{
		int mask = 255;
		int r = v & mask;
		int g = (v & (mask << 8)) >> 8;
		int b = (v & (mask << 16)) >> 16;
		std::cout << count << ". ";
		std::cout << r << " ";
		std::cout << g << " ";
		std::cout << b << "\n";
		count++;
	}
}
//INPUT: BMP CLASS OBJECTS
//CALL GPU KERNELS :
//1. d_createColorPalette - Creating colors palette
//2. d_applyPalette - Applyes palette to resize the BMP file from 24 bit to 8 bit
void gpuCall(BMP img)
{
	//GPU CREATE PALETTE
	std::vector<int> all_colors_int(img.h_all_colors.size());
	for (int i = 0; i < img.h_all_colors.size(); i++)
		all_colors_int[i] = img.h_all_colors[i].convertRGBtoINT();
	int* d_all_colors;
	int* d_color_palette;
	std::vector<int> h_palette_from_gpu(256);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc(&d_all_colors, all_colors_int.size() * sizeof(int));
	cudaMalloc(&d_color_palette, 256 * sizeof(int));
	cudaMemcpyToSymbol(color_palette, img.h_color_palette.data(), 256);
	cudaEventRecord(start);
	cudaMemcpy(d_all_colors, all_colors_int.data(), all_colors_int.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_color_palette, h_palette_from_gpu.data(), 256 * sizeof(int), cudaMemcpyHostToDevice);
	dim3 dimGrid(1);
	dim3 dimBlock;
	int num;
	if (all_colors_int.size() > 1024)
	{
		dimBlock.x = 256;
		num = ceil(double(img.h_all_colors.size()) / 256.);
	}
	else
	{
		dimBlock.x = all_colors_int.size();
		num = 1;
	}
	d_createColorPalette<<<dimGrid, dimBlock>>> (d_all_colors, d_color_palette, all_colors_int.size(), num);
	cudaDeviceSynchronize();
	cudaMemcpy(h_palette_from_gpu.data(), d_color_palette, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	cudaError_t err1 = cudaGetLastError();
	if (err1 != cudaSuccess) printf("%s ",cudaGetErrorString(err1));
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cleanPalette(h_palette_from_gpu);
	std::cout << "GPU PALETTE:\n";
	printIntPaletteToRgb(h_palette_from_gpu);
	std::cout << "\ngpu milliseconds elapsed for creating palette: " << milliseconds << '\n';
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//-------------------------------------------------------------------------------------------------------
	// APPLY PALETTE
	std::vector<UINT8> h_applyPalette_result(all_colors_int.size());
	UINT8* d_applyPalette_result;
	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1);
	cudaMalloc(&d_applyPalette_result, all_colors_int.size() * sizeof(UINT8));
	dim3 Grid(ceil(double(all_colors_int.size())/32.));
	dim3 Block(32);
	d_applyPalette<<<Grid, Block>>> (d_all_colors, d_color_palette, d_applyPalette_result, all_colors_int.size());
	cudaDeviceSynchronize();
	cudaMemcpy(h_applyPalette_result.data(), d_applyPalette_result, all_colors_int.size() * sizeof(UINT8), cudaMemcpyDeviceToHost);
	err1 = cudaGetLastError();
	if (err1 != cudaSuccess) printf("%s ", cudaGetErrorString(err1));
	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start1, stop1);
	std::cout << "gpu milliseconds elapsed for applying palette: " << milliseconds << '\n';
	cudaFree(d_all_colors);
	cudaFree(d_color_palette);
	cudaFree(d_applyPalette_result);

	BMP test;
	test.h_color_palette.resize(256);
	for (int i=0;i<h_palette_from_gpu.size(); i++)
	{

		test.h_color_palette[i] = BMP::RGB::convertINTtoRGB(h_palette_from_gpu[i]);
	}

	test.h_all_colors_resize.resize(h_applyPalette_result.size());
	test.h_all_colors_resize = h_applyPalette_result;
	test.returnColors(test.h_all_colors);
	bool check = true;
	for (int i = 0; i < all_colors_int.size(); i++)
		if (test.h_all_colors[i] != img.h_all_colors[i])
		{
			check = false;
			std::cout << i << " ";
			std::cout << (int)test.h_all_colors[i].red <<" " << (int)test.h_all_colors[i].green << " " << (int)test.h_all_colors[i].blue<<" and ";
			std::cout << (int)img.h_all_colors[i].red << " " << (int)img.h_all_colors[i].green << " " << (int)img.h_all_colors[i].blue;
			break;
		}
	std::cout << "TEST. create and apply palette on GPU and decode on CPU: ";
	if (check)
		std::cout << "ok";
	else std::cout << "ne ok";
	std::cout << "\n";
}

int main()
{
    
	//printDeviceProp();
	//BMP image("testbmp.bmp");
	//BMP image("testbmp3.bmp");
	BMP image("parrots.bmp");
	//BMP image("RAKETA.bmp");
	image.collectAllColors();
	image.h_createColorPallete();
	image.h_applyPalette();


	std::cout << "CPU PALETTE:\n";
    image.printColorPallete();
   // image.printAllColorsResize();
   // image.writeHTML();
	gpuCall(image);
	std::cout << "cpu milliseconds elapsed for creating palette: " << image.elapsed_palette.count() << "\n";
	std::cout << "cpu milliseconds elapsed for applying palette: " << image.elapsed_applying.count() << "\n";
}
