#pragma once

#include <cassert>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Fluxel.h"
#include "Object.h"
#include "Logger.h"

NAMESPACE_BEGIN(fluxel)

#define CUDA_ASSERT(cudaCall)                                                                                          \
	do                                                                                                                 \
	{                                                                                                                  \
		cudaError_t __cudaError = (cudaCall);                                                                          \
		if (__cudaError != cudaSuccess)                                                                                \
		{                                                                                                              \
			Log(Error, "CUDA error: %s at %s:%d", cudaGetErrorString(__cudaError), __FILE__, __LINE__);                \
			assert(false);                                                                                             \
		}                                                                                                              \
	} while (0);

#define CUDA_CHECK(cudaCall)                                                                                           \
	do                                                                                                                 \
	{                                                                                                                  \
		CUresult __cuResult = (cudaCall);                                                                              \
		if (__cuResult != CUDA_SUCCESS)                                                                                \
		{                                                                                                              \
			Log(Error, "CUDA error: %d at %s:%d", __cuResult, __FILE__, __LINE__);                                     \
			assert(false);                                                                                             \
		}                                                                                                              \
	} while (0);

class CUDAContext {
public:
	CUDAContext();
    ~CUDAContext();

	[[nodiscard]] cudaStream_t getStream() const { return m_stream; }
	[[nodiscard]] CUcontext getContext() const { return m_context; }
	[[nodiscard]] cudaDeviceProp getDeviceProps() const { return m_deviceProps; }
	[[nodiscard]] CUdevice getDevice() const { return m_device; }

private:
  	cudaStream_t m_stream;
	CUcontext m_context;
	CUdevice m_device;
	cudaDeviceProp m_deviceProps;
};

// TODO: Implement this interface to abstract the execution provider.
class IExecutionProvider {
public:
};


NAMESPACE_END(fluxel)
