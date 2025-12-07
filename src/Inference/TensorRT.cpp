#include <string>
#include <cassert>
#include "NvInferRuntime.h"
#include "NvOnnxParser.h"

#include "Inference/TensorRT.h"
#include "Logger.h"
#include "nvrhi/common/resource.h"

NAMESPACE_BEGIN(fluxel)

void TRTLogger::log(nvinfer1::ILogger::Severity severity,
                    const char *msg) noexcept {
	auto convertSeverity = [](nvinfer1::ILogger::Severity severity) -> Log::Level {
		switch (severity) {
			case nvinfer1::ILogger::Severity::kVERBOSE: return Log::Level::Debug;
			case nvinfer1::ILogger::Severity::kINFO: return Log::Level::Info;
			case nvinfer1::ILogger::Severity::kWARNING: return Log::Level::Warning;
			case nvinfer1::ILogger::Severity::kERROR: return Log::Level::Error;
			case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return Log::Level::Error;
			default: return Log::Level::None;
		}
	};

	Logger::log(convertSeverity(severity), "[TRT-RTX] " + std::string(msg));
}

float getTensorElementSize(nvinfer1::DataType dataType) {
	switch (dataType) {
		case nvinfer1::DataType::kFLOAT: return 4.0f;
		case nvinfer1::DataType::kHALF: return 2.0f;
		case nvinfer1::DataType::kINT8: return 1.0f;
		case nvinfer1::DataType::kINT32: return 4.0f;
		case nvinfer1::DataType::kBOOL: return 4.0f;
		case nvinfer1::DataType::kUINT8: return 1.0f;
		case nvinfer1::DataType::kINT64: return 8.0f;
		case nvinfer1::DataType::kBF16: return 2.0f;
		case nvinfer1::DataType::kFP8: return 1.0f;
		case nvinfer1::DataType::kFP4: return 0.5f;
		default: assert(false && "Unsupported data type");
		return 0.0f;
	}
}

float TensorDescriptor::elementSize() const {
	return getTensorElementSize(this->type);
}

TensorRTExecutionProvider::TensorRTExecutionProvider(donut::app::DeviceManager *deviceManager)
    : CommonDeviceObject(deviceManager) {}

std::shared_ptr<SharedTensor> TensorRTExecutionProvider::getTensor(const std::string &name) {
	return m_tensors[name];
}

nvrhi::BufferHandle TensorRTExecutionProvider::getTensorBuffer(const std::string &name) {
	return m_tensors[name]->buffer;
}

std::shared_ptr<SharedTensor> TensorRTExecutionProvider::createSharedTensor(
    const TensorDescriptor &descriptor) {

  	auto sharedTensor = std::make_shared<SharedTensor>();
	sharedTensor->descriptor = descriptor;
	sharedTensor->cudaPtr = nullptr;
	sharedTensor->cudaExternalMemory = nullptr;
    sharedTensor->buffer = nullptr;

    nvrhi::BufferDesc bufferDesc;
	bufferDesc.byteSize = descriptor.shape.size() * descriptor.elementSize();
	bufferDesc.debugName = descriptor.name;
	bufferDesc.canHaveUAVs = true;
	bufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
	bufferDesc.keepInitialState = false;
	// Setup shared resource flags for CUDA external memory export.
    bufferDesc.sharedResourceFlags = nvrhi::SharedResourceFlags::Shared;

    // Create a buffer for the shared tensor.
    sharedTensor->buffer = GetDevice()->createBuffer(bufferDesc);

    // Export the buffer to CUDA external memory.
    auto sharedHandle = sharedTensor->buffer->getNativeObject(nvrhi::ObjectTypes::SharedHandle);

	return sharedTensor;
}

bool TensorRTExecutionProvider::load(std::filesystem::path onnxPath) {

	if (!(onnxPath.extension() == ".onnx")) {
		logError("[TensorRT Runner] Invalid ONNX file: " + onnxPath.string());
		return false;
        }

    TRTLogger logger;

	auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
	if (!builder) {
		logError("[TensorRT Runner] Failed to create builder!");
		return false;
	}

	auto builderConfig = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	if (!builderConfig) {
		logError("[TensorRT Runner] Failed to create builder config!");
		return false;
	}

	// TRT-RTX only supports strongly typed networks, explicitly specify this to avoid warning.
	nvinfer1::NetworkDefinitionCreationFlags flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);

	auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
	if (!network) {
		logError("[TensorRT Runner] Failed to create network!");
		return false;
	}

	auto parser = nvonnxparser::createParser(*network, logger);
	if (!parser) {
		logError("[TensorRT Runner] Failed to create parser!");
                return false;
        }

	if (!parser->parseFromFile(onnxPath.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE))) {
		logError("[TensorRT Runner] Failed to parse ONNX file!");
		return false;
        }

    auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *builderConfig));

    logInfo("[TensorRT Runner] Successfully built the serialized engine. Engine size: " + std::to_string(serializedEngine->size()) + " bytes.");

    m_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!m_runtime) {
        logError("[TensorRT Runner] Failed to create runtime!");
        return false;
    }

    // Deserialize the engine.
	m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));
    if (!m_engine) {
    	logError("[TensorRT Runner] Failed to create inference engine!");
        return false;
    }

    // Optional settings to configure the behavior of the inference runtime.
    auto runtimeConfig = std::unique_ptr<nvinfer1::IRuntimeConfig>(m_engine->createRuntimeConfig());
    if (!runtimeConfig) {
        logError("[TensorRT Runner] Failed to create runtime config!");
        return false;
    }

    // Create an engine execution context out of the deserialized engine.
    // TRT-RTX performs "Just-in-Time" (JIT) optimization here, targeting the current GPU.
    m_executionContext = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext(runtimeConfig.get()));
    if (!m_executionContext) {
        logError("[TensorRT Runner] Failed to create execution context!");
        return false;
    }

    // Parse input and output tensor names from the network.
    for (int i = 0; i < m_engine->getNbIOTensors(); i++) {
		// Parse tensor name, shape and element size.
      	TensorDescriptor tensor;
		tensor.name = m_engine->getIOTensorName(i);

		auto dims = m_engine->getTensorShape(tensor.name.c_str());
		for (int dim = 0; dim < dims.nbDims; dim++) {
			tensor.shape.push_back(dims.d[dim] < 0 ? 1U : dims.d[dim]);
		}

		tensor.type = m_engine->getTensorDataType(tensor.name.c_str());

        switch (m_engine->getTensorIOMode(tensor.name.c_str())) {
        case nvinfer1::TensorIOMode::kINPUT:
			Log(Debug, "[TensorRT Runner] Parsed input tensor: %s", tensor.name.c_str());
			m_inputDescriptors[tensor.name] = tensor;
			break;
        case nvinfer1::TensorIOMode::kOUTPUT:
			Log(Debug, "[TensorRT Runner] Parsed output tensor: %s", tensor.name.c_str());
			m_outputDescriptors[tensor.name] = tensor;
			break;
        default: break;
		}
    }

    // Create buffers for IO tensors and setup their addresses in the execution context.
    for (auto &[name, descriptor] : m_inputDescriptors) {
		m_tensors[name] = createSharedTensor(descriptor);
    }

	return true;
}

NAMESPACE_END(fluxel)
