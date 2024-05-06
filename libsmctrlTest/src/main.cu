#include <stdio.h>
#include <stdlib.h>

#include "../libsmctrl/libsmctrl.h"

#include <chrono>

#define CUDALOOPS 5000
constexpr int TOTAL_TPCs = 19; // 19 TPCs in our 3060 Ti

// This macro determines on how many SMs our GPU kernel
// will be optimal on. Each TPC contains 2 SMs in our 3060Ti
// that means each partition can handle ~2048 parallel ops per cycle.
// So if you divide OPTIMAL_WIDTH/2048 = 8, meaning this kernel
// will be optimal on 8 SMs. Feel free to change this value.
#define OPTIMAL_WIDTH 16384

unsigned long long my_mask = 0;

__global__ void vecAdd_kernel(double* a, double* b, double* c, int n) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) {
		for (int i = 0; i < CUDALOOPS; ++i) {
			c[id] = c[id] + a[id] + b[id];
			c[id] = c[id] + a[id];
			c[id] = c[id] + b[id];
		}
	}
}

void vecAdd_cuda(double* a, double* b, double* c, int n) {
	double* da, * db, * dc;
	size_t bytes = n * sizeof(double);
	cudaMalloc(&da, bytes);
	cudaMalloc(&db, bytes);
	cudaMalloc(&dc, bytes);

	cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

	int blockSize = 1024, gridSize;
	gridSize = (int)ceil((float)n / blockSize);

	cudaStream_t myStream;
	cudaStreamCreate(&myStream);

	libsmctrl_set_stream_mask(myStream, my_mask);

	vecAdd_kernel << <gridSize, blockSize, 0, myStream >> > (da, db, dc, n);

	cudaMemcpy(dc, c, bytes, cudaMemcpyDeviceToHost);
	cudaStreamSynchronize(myStream);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
}

int main(int argc, char* argv[]) {
	double* a, * b, * c;

	int n = OPTIMAL_WIDTH;
	size_t nBytes = n * sizeof(double);
	a = (double*)malloc(nBytes);
	b = (double*)malloc(nBytes);
	c = (double*)malloc(nBytes);

	for (int i = 0; i < n; ++i) {
		a[i] = sin(i) * sin(i);
		b[i] = cos(i) * cos(i);
	}

	// First, we do a warmup kernel launch, because the first kernel takes an eternity
	vecAdd_cuda(a, b, c, n);

	for (int i = 0; i < TOTAL_TPCs; ++i) {
		auto startTime = std::chrono::high_resolution_clock::now();
		vecAdd_cuda(a, b, c, n);
		auto endTime = std::chrono::high_resolution_clock::now();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
		printf("%d Partitions: %ldms\n", TOTAL_TPCs - i, ms);
		my_mask <<= 1;
		my_mask |= 1;
	}
	return 0;
}
