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

#include <Inference/NeuralInferencePass.h>
#include <Inference/TensorRT.h>
#include <Render/RenderPass.h>
#include <Utils/DeviceUtils.h>
#include <Utils/FileSystem.h>
#include <Logger.h>
#include "plugins/BasicRenderer/Renderer.h"

using namespace fluxel;

class ModelRunnerUI : public UIRenderPass {
public:
	ModelRunnerUI(donut::app::DeviceManager *deviceManager) : UIRenderPass(deviceManager) {}
	~ModelRunnerUI() {}

	void buildUI() override {
		ImGui::Text("Model Runner");
        ImGui::Text("Renderer: %s", GetDeviceManager()->GetRendererString());
		double frameTime = GetDeviceManager()->GetAverageFrameTimeSeconds();
        if (frameTime > 0.0)
            ImGui::Text("%.3f ms/frame (%.1f FPS)", frameTime * 1e3, 1.0 / frameTime);
	}
};

int main(int argc, char **argv) {
	TRTLogger logger;

	// Currently only Vulkan is supported for interoperation between CUDA and Graphics.
    nvrhi::GraphicsAPI graphicsAPI = nvrhi::GraphicsAPI::VULKAN;

    std::shared_ptr<donut::app::DeviceManager> deviceManager(donut::app::DeviceManager::Create(graphicsAPI));

	donut::app::DeviceCreationParameters deviceParams;

	deviceParams.backBufferWidth = 2048;
	deviceParams.backBufferHeight = 1152;

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

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, "Model Runner")) {
		Log(Fatal, "Failed to create a window device and swap chain.");
		return EXIT_FAILURE;
    }

	{
		auto uiRenderer = std::make_shared<ModelRunnerUI>(deviceManager.get());
		auto neuralInferencePass = std::make_shared<NeuralInferencePass>(deviceManager.get());
		auto basicRenderer = std::make_shared<BasicRenderer>(deviceManager.get(), "");

		uiRenderer->AddRenderPass(basicRenderer);
		uiRenderer->AddRenderPass(neuralInferencePass);

		deviceManager->AddRenderPassToBack(basicRenderer.get());
		deviceManager->AddRenderPassToBack(neuralInferencePass.get());
		deviceManager->AddRenderPassToBack(uiRenderer.get());

		deviceManager->RunMessageLoop();

		deviceManager->RemoveRenderPass(neuralInferencePass.get());
		deviceManager->RemoveRenderPass(uiRenderer.get());
	}

    deviceManager->Shutdown();
    deviceManager.reset();

    return EXIT_SUCCESS;
}
