#include <driver_types.h>
#include <string>
#include <cassert>
#include <filesystem>
#include <system_error>
#include <numeric>
#include "NvInferRuntime.h"
#include "NvOnnxParser.h"

#include "TensorRT.h"
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

float TensorDescriptor::elementSize() const { return getTensorElementSize(this->type); }

size_t TensorDescriptor::elementCount() const { return std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>()); }

size_t TensorDescriptor::byteSize() const { return elementCount() * elementSize(); }

TensorRTExecutionProvider::TensorRTExecutionProvider(nvrhi::IDevice *device)
    : CommonDeviceObject(device) {}

std::shared_ptr<SharedTensor> TensorRTExecutionProvider::getTensor(const std::string &name) {
	if (m_tensors.find(name) == m_tensors.end()) {
		Log(Warning, "[TensorRT Runner] Tensor not found: %s", name.c_str());
		return nullptr;
	}
	return m_tensors[name];
}

nvrhi::IBuffer* TensorRTExecutionProvider::getTensorBuffer(const std::string &name) {
	return getTensor(name)->buffer.Get();
}

std::shared_ptr<SharedTensor> TensorRTExecutionProvider::createSharedTensor(
    const TensorDescriptor &descriptor) {

  	auto sharedTensor = std::make_shared<SharedTensor>();
	sharedTensor->descriptor = descriptor;

    nvrhi::BufferDesc bufferDesc = {};
	bufferDesc.format			 = nvrhi::Format::UNKNOWN;
	bufferDesc.structStride		 = static_cast<uint32_t>(descriptor.elementSize());
	bufferDesc.byteSize			 = static_cast<size_t>(descriptor.elementCount() * descriptor.elementSize());
	bufferDesc.debugName		 = descriptor.name;
	bufferDesc.canHaveUAVs		 = true;
	bufferDesc.canHaveRawViews	 = false;
	bufferDesc.canHaveTypedViews = false;
	bufferDesc.initialState		 = nvrhi::ResourceStates::UnorderedAccess;
	bufferDesc.keepInitialState	 = true;
	// Setup shared resource flags for CUDA external memory export.
    bufferDesc.sharedResourceFlags = nvrhi::SharedResourceFlags::Shared;

    // Create a buffer for the shared tensor.
    sharedTensor->buffer = getDevice()->createBuffer(bufferDesc);

    // Export the buffer to CUDA external memory.
    HANDLE sharedHandle =
        sharedTensor->buffer->getNativeObject(nvrhi::ObjectTypes::SharedHandle);

    if (getDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN) {
    	// Import the handle for a Vulkan buffer to CUDA as an external memory.
		cudaExternalMemoryHandleDesc handleDesc = {};
        handleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        handleDesc.handle.win32.name = nullptr;
        handleDesc.handle.win32.handle = sharedHandle;
        handleDesc.size = bufferDesc.byteSize;

        CUDA_ASSERT(cudaImportExternalMemory(&sharedTensor->cudaExternalMemory,
                                             &handleDesc));

        // Map the CUDA external memory to device pointer.
        cudaExternalMemoryBufferDesc cudaBufferDesc = {};
        cudaBufferDesc.flags = 0;
        cudaBufferDesc.offset = 0;
        cudaBufferDesc.size = bufferDesc.byteSize;

        CUDA_ASSERT(cudaExternalMemoryGetMappedBuffer(&sharedTensor->cudaPtr, sharedTensor->cudaExternalMemory, &cudaBufferDesc));

    } else {
		Log(Fatal, "[TensorRT Runner] Inference is not implemented for this graphics API: %d", getDevice()->getGraphicsAPI());
	}

	return sharedTensor;
}

bool TensorRTExecutionProvider::load(std::filesystem::path onnxPath) {

	// Destroy any existing engine and execution context created by the previous load call.
	// The execution context should be destroyed before any engine objects that it created.
	m_executionContext.reset();
	m_engine.reset();
	m_runtime.reset();

	if (!(onnxPath.extension() == ".onnx")) {
		logError("[TensorRT Runner] Invalid ONNX file: " + onnxPath.string());
		return false;
        }

	TRTLogger logger;
	std::string enginePath = onnxPath.string() + ".engine";

	if (!std::filesystem::exists(enginePath)) {
		Log(Info, "TensorRT Runner] Building the serialized engine using ONNX file: %s",
			onnxPath.c_str());

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

		// TensorRT resolves external ONNX data relative to the process CWD.
		// Ensure the CWD is the ONNX folder during parsing, and normalize path
		// separators so TensorRT can locate the `.onnx.data` file on Windows.
		const std::filesystem::path onnxAbsolutePath = std::filesystem::absolute(onnxPath);
		const std::filesystem::path onnxDir = onnxAbsolutePath.parent_path();

		struct CurrentPathGuard {
			std::filesystem::path previous;
			explicit CurrentPathGuard(const std::filesystem::path &next) : previous(std::filesystem::current_path()) {
				std::filesystem::current_path(next);
			}
			~CurrentPathGuard() {
				std::error_code ec;
				std::filesystem::current_path(previous, ec);
			}
		} cwdGuard(onnxDir);

		const std::string onnxPathStr = onnxAbsolutePath.generic_string(); // use forward slashes
		if (!parser->parseFromFile(onnxPathStr.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE))) {
			logError("[TensorRT Runner] Failed to parse ONNX file!");
			return false;
		}

		auto optimizationProfile = builder->createOptimizationProfile();
		// Handle dynamic shapes
		for (int i = 0; i < network->getNbInputs(); i++) {
			auto input = network->getInput(i);
			std::string inputName = input->getName();
			auto inputShape = input->getDimensions();

			if (inputShape.d[0] == -1) {
				nvinfer1::Dims targetShape = inputShape;
				targetShape.d[0]		   = 1;
				if (inputShape.nbDims == 4) {
					targetShape.d[1] = 3;
					targetShape.d[2] = 1152;
					targetShape.d[3] = 2048;
				}
				optimizationProfile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kMIN, targetShape);
				optimizationProfile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kOPT, targetShape);
				optimizationProfile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kMAX, targetShape);
			}
		}
		builderConfig->addOptimizationProfile(optimizationProfile);

		auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *builderConfig));

		Log(Success, "[TensorRT Runner] Successfully built the serialized engine. Engine size: %lu bytes.",
			serializedEngine->size());

		Log(Success, "[TensorRT Runner] Caching serialized engine to file: %s", enginePath.c_str());
		std::ofstream engineFile(enginePath, std::ios::binary);
		engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
		engineFile.close();

	} else {
		Log(Success, "[TensorRT Runner] Engine file already exists. Loading from file: %s",
			enginePath.c_str());
	}

	// Load the serialized engine from file and deserialize it.
	std::ifstream engineFile(enginePath, std::ios::binary);
	std::vector<char> serializedEngine(std::istreambuf_iterator<char>(engineFile), {});
	engineFile.close();

    m_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!m_runtime) {
        logError("[TensorRT Runner] Failed to create runtime!");
        return false;
    }

    // Deserialize the engine.
	m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(serializedEngine.data(), serializedEngine.size()));
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
		auto tensor = createSharedTensor(descriptor);

        m_tensors[name] = tensor;
        m_executionContext->setTensorAddress(name.c_str(), tensor->cudaPtr);

        // Set the shape for current input in case it is built with dynamic shapes.
        nvinfer1::Dims dims;
        dims.nbDims = descriptor.shape.size();
        for (size_t i = 0; i < descriptor.shape.size(); i++) {
            dims.d[i] = descriptor.shape[i];
        }
        m_executionContext->setInputShape(name.c_str(), dims);
    }

    for (auto &[name, descriptor] : m_outputDescriptors) {
      auto tensor = createSharedTensor(descriptor);

      m_tensors[name] = tensor;
      m_executionContext->setTensorAddress(name.c_str(), tensor->cudaPtr);
    }

    // Create a CUDA context for the execution provider.
    m_cudaContext = std::make_unique<CUDAContext>();

	return true;
}

void TensorRTExecutionProvider::inference() {
	// TODO: Implement asynchronous inference with multiple on-flight inference requests.
	// TODO: Check whether we can set the dimension at the dynamic axis as 1.

    m_executionContext->enqueueV3(m_cudaContext->getStream());
}

NAMESPACE_END(fluxel)
