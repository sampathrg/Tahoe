#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cub/cub.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#include <cuda_profiler_api.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>

//#define _DEBUG_ 1

// FIL_TPB is the number of threads per block to use with FIL kernels
const int FIL_TPB = 256;

/** check for cuda runtime API errors and assert accordingly */
#define CUDA_CHECK(call)                                                 \
do {                                                                   \
    cudaError_t status = call;                                           \
    if(status != cudaSuccess)                                           \
		printf("FAIL: call='%s'. Reason:%s\n", #call, \
           cudaGetErrorString(status));                                  \
} while (0)

/** cuda malloc */
template <typename Type>
void allocate(Type*& ptr, size_t len, bool setZero = true) {
  CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(Type) * len));
  if (setZero) CUDA_CHECK(cudaMemset(ptr, 0, sizeof(Type) * len));
}

/** performs a host to device copy */
template <typename Type>
void updateDevice(Type* dPtr, const Type* hPtr, size_t len,
                  cudaStream_t stream) {
  copy(dPtr, hPtr, len, stream);
}

/** performs a device to host copy */
template <typename Type>
void updateHost(Type* hPtr, const Type* dPtr, size_t len, cudaStream_t stream) {
	  copy(hPtr, dPtr, len, stream);
}

template <typename Type>
void copy(Type* dst, const Type* src, size_t len, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, len * sizeof(Type), cudaMemcpyDefault, stream));
}

template <typename IntType>
IntType ceildiv(IntType a, IntType b) {
    return (a + b - 1) / b;
}

__global__ void nan_kernel(float* data, const bool* mask, int len, float nan) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= len) return;
  if (!mask[tid]) data[tid] = nan;
}

__global__ void printf_float_GPU(float* data, int num_rows)
{
	for(int i=0; i<num_rows; i++)
		printf("%f, ", data[i]);
	printf("\n");
}

__global__ void printf_int_GPU(int* data, int num_rows)
{
	for(int i=0; i<num_rows; i++)
		printf("%d, ", data[i]);
	printf("\n");
}

void printf_float_CPU(float* data, int num_rows)
{
	for(int i=0; i<num_rows; i++)
		printf("%8.4f, ", data[i]);
	printf("\n");
}

void printf_int_CPU(int* data, int num_rows)
{
	for(int i=0; i<num_rows; i++)
		printf("%d, ", data[i]);
	printf("\n");
}

void printf_bool_CPU(bool* data, int num_rows)
{
	for(int i=0; i<num_rows; i++)
		printf("%d, ", data[i]);
	printf("\n");
}

__global__ void compare_GPU(float* data1, float* data2, int length)
{
	bool flag = true;
	for(int i=0; i<length; i++)
	{
		if( (data1[i]-data2[i])>1e-3 || (data1[i]-data2[i])<-1e-3 )
			flag = false;
	}
	if(flag == true)
		printf("Results are correct\n");
	else
		printf("Results are incorrect\n");

}

__device__ int ceildiv_dev(int a, int b) {
    return (int)((a + b - 1) / b);
}

