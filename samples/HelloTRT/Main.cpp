#include <cassert>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <vector>

#include <NvOnnxParser.h>

#include <nvrhi/nvrhi.h>
#include <nvrhi/utils.h>

#include <donut/app/ApplicationBase.h>

#include "plugins/NeuralInference/TensorRT.h"
#include <Utils/DeviceUtils.h>
#include <Utils/FileSystem.h>
#include <Logger.h>

using namespace fluxel;

template <typename T>
void printBuffer(std::ostream& os, const std::string& name, const T& buffer)
{
	os << name << ": ";
	for (const auto& value : buffer) {
		os << value << " ";
	}
	os << std::endl;
}

int main(int argc, char **argv) {
	TRTLogger logger;

	// Currently only Vulkan is supported for interoperation between CUDA and Graphics.
    nvrhi::GraphicsAPI graphicsAPI = nvrhi::GraphicsAPI::VULKAN;

    std::shared_ptr<donut::app::DeviceManager> deviceManager(donut::app::DeviceManager::Create(graphicsAPI));

    donut::app::DeviceCreationParameters deviceParams;

    deviceParams.requiredVulkanDeviceExtensions.push_back("VK_KHR_external_memory_win32");
#ifdef FLUXEL_DEBUG_BUILD
    deviceParams.enableDebugRuntime = true;
    deviceParams.enableNvrhiValidationLayer = true;
#endif

    if (!deviceManager->CreateInstance(deviceParams)) {
		Log(Fatal, "Failed to create a device manager instance.");
		return EXIT_FAILURE;
    }

    SelectPreferredAdapter(deviceManager.get(), deviceParams);

    if (!deviceManager->CreateHeadlessDevice(deviceParams)) {
		Log(Fatal, "Failed to create a headless device.");
		return EXIT_FAILURE;
    }

    Log(Info, "Initialized graphics device using %s API with %s.", nvrhi::utils::GraphicsAPIToString(graphicsAPI), deviceManager->GetRendererString());

    // Create a TensorRT runner.
	{
		auto runner = std::make_unique<TensorRTExecutionProvider>(deviceManager->GetDevice());

		std::filesystem::path onnxModelPath = file::projectDir() / "assets/data/HelloTRT.onnx";
		if (!runner->load(onnxModelPath)) {
			Log(Fatal, "Failed to load the ONNX model.");
			return EXIT_FAILURE;
		}

		auto input = runner->getTensor("input");
		auto output = runner->getTensor("output");

		std::vector<float> inputData(3);
		std::vector<float> outputData(2);

		for (int i = 0; i < 5; i++) {
			std::fill(inputData.begin(), inputData.end(), static_cast<float>(i));

			CUDA_ASSERT(cudaMemcpy(input->cudaPtr, inputData.data(),
								   inputData.size() * sizeof(float), cudaMemcpyHostToDevice));

			runner->inference();

			CUDA_ASSERT(cudaMemcpy(outputData.data(), output->cudaPtr,
								   outputData.size() * sizeof(float), cudaMemcpyDeviceToHost));
			CUDA_ASSERT(cudaDeviceSynchronize());

			printBuffer(std::cout, "Input", inputData);
			printBuffer(std::cout, "Output", outputData);
		}
	}

    Log(Success, "Successfully ran the network.");

    deviceManager->Shutdown();
    return EXIT_SUCCESS;
}
