#pragma once

#include <filesystem>
#include <memory>
#include <vector>
#include <unordered_map>

#include <nvrhi/nvrhi.h>
#include <donut/app/DeviceManager.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "NvInfer.h"
#include "NvInferRuntime.h"

#include "Fluxel.h"
#include "Object.h"
#include "Inference/Common.h"

NAMESPACE_BEGIN(fluxel)

class TRTLogger : public nvinfer1::ILogger {
public:
	TRTLogger() = default;
	~TRTLogger() override = default;

	void log(nvinfer1::ILogger::Severity severity,
				const char *msg) noexcept override;
private:
};

struct TensorDescriptor {
	std::string name;
	std::vector<size_t> shape;
    nvinfer1::DataType type;

	float elementSize() const;
	size_t elementCount() const;
	size_t byteSize() const;
};

struct SharedTensor {
	void *cudaPtr = nullptr;
	cudaExternalMemory_t cudaExternalMemory = nullptr;
	nvrhi::BufferHandle buffer = nullptr;
    TensorDescriptor descriptor;
};

class TensorRTExecutionProvider : public IExecutionProvider, public CommonDeviceObject {
public:
  	explicit TensorRTExecutionProvider(nvrhi::IDevice *device);
  	~TensorRTExecutionProvider() override = default;

    bool load(std::filesystem::path onnxPath);
	void inference();

	[[nodiscard]] const std::unordered_map<std::string, TensorDescriptor> &getInputDescriptors() const { return m_inputDescriptors; }
	[[nodiscard]] const std::unordered_map<std::string, TensorDescriptor> &getOutputDescriptors() const { return m_outputDescriptors; }

    [[nodiscard]] std::shared_ptr<SharedTensor> getTensor(const std::string &name);
    [[nodiscard]] nvrhi::IBuffer* getTensorBuffer(const std::string &name);

private:
	std::shared_ptr<SharedTensor> createSharedTensor(const TensorDescriptor &descriptor);

	// TensorRT
	std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
	std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
	std::unique_ptr<nvinfer1::IExecutionContext> m_executionContext = nullptr;

	// CUDA and synchronization
	cudaExternalSemaphore_t m_cudaSemaphore = nullptr;
	std::unique_ptr<CUDAContext> m_cudaContext;

    // IO tensors and descriptors
	std::unordered_map<std::string, TensorDescriptor> m_inputDescriptors;
	std::unordered_map<std::string, TensorDescriptor> m_outputDescriptors;
	std::unordered_map<std::string, std::shared_ptr<SharedTensor>> m_tensors;
};

NAMESPACE_END(fluxel)
