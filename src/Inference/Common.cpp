#include "Inference/Common.h"
#include <cuda.h>

NAMESPACE_BEGIN(fluxel)

// TODO: Ensure that the selected CUDA device is the same as the Graphics API device.
CUDAContext::CUDAContext() {
    // initialize cuda
  	CUDA_ASSERT(cudaFree(0));
  	int numDevices = 0;
	CUDA_ASSERT(cudaGetDeviceCount(&numDevices));
	if (numDevices == 0)
		Log(Fatal, "No CUDA capable devices found!");
	Log(Info, "Found " + std::to_string(numDevices) + " CUDA device(s).");

	// set up context
	const int deviceID = 0;
	CUDA_ASSERT(cudaSetDevice(deviceID));
	CUDA_ASSERT(cudaStreamCreate(&m_stream));
	CUDA_ASSERT(cudaGetDevice(&m_device));

	cudaGetDeviceProperties(&m_deviceProps, deviceID);
	Log(Success, "Fluxel is running on %s", m_deviceProps.name);
	if (!m_deviceProps.concurrentManagedAccess)
		Log(Debug, "Concurrent access of managed memory is not supported.");

    CUctxCreateParams createParams = {nullptr, 0, nullptr};
	CUDA_CHECK(cuCtxCreate_v4(&m_context, &createParams, 0, m_device));
}

CUDAContext::~CUDAContext() {
	CUDA_ASSERT(cudaStreamDestroy(m_stream));
	CUDA_CHECK(cuCtxDestroy(m_context));
}

NAMESPACE_END(fluxel)
