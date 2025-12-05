#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvInferRuntime.h"

#include "Fluxel.h"


#define CUDA_ASSERT(cudaCall)                                                                                          \
	do                                                                                                                 \
	{                                                                                                                  \
		cudaError_t __cudaError = (cudaCall);                                                                          \
		if (__cudaError != cudaSuccess)                                                                                \
		{                                                                                                              \
			std::cerr << "CUDA error: " << cudaGetErrorString(__cudaError) << " at " << __FILE__ << ":" << __LINE__    \
					<< std::endl;                                                                                    \
			assert(false);                                                                                             \
		}                                                                                                              \
	} while (0)

NAMESPACE_BEGIN(fluxel)

class TRTLogger : public nvinfer1::ILogger {
public:
	TRTLogger() = default;
	~TRTLogger() override = default;

	void log(nvinfer1::ILogger::Severity severity,
				const char *msg) noexcept override;
private:
};

NAMESPACE_END(fluxel)