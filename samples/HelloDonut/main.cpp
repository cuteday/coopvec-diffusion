#include <string>
#include <vector>
#include <memory>
#include <chrono>

#include <donut/core/vfs/VFS.h>
#include <donut/core/log.h>
#include <donut/core/string_utils.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/ConsoleInterpreter.h>
#include <donut/engine/ConsoleObjects.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/Scene.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/render/BloomPass.h>
#include <donut/render/CascadedShadowMap.h>
#include <donut/render/DeferredLightingPass.h>
#include <donut/render/DepthPass.h>
#include <donut/render/DrawStrategy.h>
#include <donut/render/ForwardShadingPass.h>
#include <donut/render/GBuffer.h>
#include <donut/render/GBufferFillPass.h>
#include <donut/render/LightProbeProcessingPass.h>
#include <donut/render/PixelReadbackPass.h>
#include <donut/render/SkyPass.h>
#include <donut/render/SsaoPass.h>
#include <donut/render/TemporalAntiAliasingPass.h>
#include <donut/render/ToneMappingPasses.h>
#include <donut/render/MipMapGenPass.h>
#include <donut/render/DLSS.h>
#include <donut/app/ApplicationBase.h>
#include <donut/app/UserInterfaceUtils.h>
#include <donut/app/Camera.h>
#include <donut/app/DeviceManager.h>
#include <donut/app/imgui_console.h>
#include <donut/app/imgui_renderer.h>
#include <nvrhi/utils.h>
#include <nvrhi/common/misc.h>

#ifdef DONUT_WITH_TASKFLOW
#include <taskflow/taskflow.hpp>
#endif

#include "Fluxel.h"
#include "Render/RenderPass.h"
#include "Logger.h"
#include "Utils/DeviceUtils.h"

#include "plugins/BasicRenderer/Renderer.h"

using namespace fluxel;

bool g_PrintFormats = false;

bool ProcessCommandLine(int argc, const char* const* argv, donut::app::DeviceCreationParameters& deviceParams, std::string& sceneName)
{
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-width"))
        {
            deviceParams.backBufferWidth = std::stoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-height"))
        {
            deviceParams.backBufferHeight = std::stoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-fullscreen"))
        {
            deviceParams.startFullscreen = true;
        }
        else if (!strcmp(argv[i], "-debug"))
        {
            deviceParams.enableDebugRuntime = true;
            deviceParams.enableNvrhiValidationLayer = true;
        }
        else if (!strcmp(argv[i], "-no-vsync"))
        {
            deviceParams.vsyncEnabled = false;
        }
        else if (!strcmp(argv[i], "-print-formats"))
        {
            g_PrintFormats = true;
        }
        else if (argv[i][0] != '-')
        {
            sceneName = argv[i];
        }
    }

    return true;
}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nCmdShow)
{
    nvrhi::GraphicsAPI api = donut::app::GetGraphicsAPIFromCommandLine(__argc, __argv);
#else //  _WIN32
int main(int __argc, const char* const* __argv)
{
    nvrhi::GraphicsAPI api = nvrhi::GraphicsAPI::VULKAN;
#endif //  _WIN32

    donut::app::DeviceCreationParameters deviceParams;

    deviceParams.backBufferWidth = 1920;
    deviceParams.backBufferHeight = 1080;
    deviceParams.swapChainSampleCount = 1;
    deviceParams.swapChainBufferCount = 3;
    deviceParams.startFullscreen = false;
    deviceParams.vsyncEnabled = true;
    deviceParams.enablePerMonitorDPI = true;
    deviceParams.supportExplicitDisplayScaling = true;

#if DONUT_WITH_DLSS && DONUT_WITH_VULKAN
    if (api == nvrhi::GraphicsAPI::VULKAN)
    {
        donut::render::DLSS::GetRequiredVulkanExtensions(
            deviceParams.optionalVulkanInstanceExtensions,
            deviceParams.optionalVulkanDeviceExtensions);
    }
#endif

    std::string sceneName;
    if (!ProcessCommandLine(__argc, __argv, deviceParams, sceneName))
    {
        fluxel::logError("Failed to process the command line.");
        return 1;
    }

    donut::app::DeviceManager* deviceManager = donut::app::DeviceManager::Create(api);
    const char* apiString = nvrhi::utils::GraphicsAPIToString(deviceManager->GetGraphicsAPI());

    std::string windowTitle = "Donut Feature Demo (" + std::string(apiString) + ")";

    if (deviceManager->CreateInstance(deviceParams))
    {
        SelectPreferredAdapter(deviceManager, deviceParams);
    }

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, windowTitle.c_str()))
	{
        fluxel::logError("Cannot initialize a %s graphics device with the requested parameters", apiString);
		return 1;
	}

    if (g_PrintFormats)
    {
        for (uint32_t format = 0; format < (uint32_t)nvrhi::Format::COUNT; format++)
        {
            auto support = deviceManager->GetDevice()->queryFormatSupport((nvrhi::Format)format);
            const auto& formatInfo = nvrhi::getFormatInfo((nvrhi::Format)format);

            char features[13];
            features[ 0] = (support & nvrhi::FormatSupport::Buffer) != 0         ? 'B' : '.';
            features[ 1] = (support & nvrhi::FormatSupport::IndexBuffer) != 0    ? 'I' : '.';
            features[ 2] = (support & nvrhi::FormatSupport::VertexBuffer) != 0   ? 'V' : '.';
            features[ 3] = (support & nvrhi::FormatSupport::Texture) != 0        ? 'T' : '.';
            features[ 4] = (support & nvrhi::FormatSupport::DepthStencil) != 0   ? 'D' : '.';
            features[ 5] = (support & nvrhi::FormatSupport::RenderTarget) != 0   ? 'R' : '.';
            features[ 6] = (support & nvrhi::FormatSupport::Blendable) != 0      ? 'b' : '.';
            features[ 7] = (support & nvrhi::FormatSupport::ShaderLoad) != 0     ? 'L' : '.';
            features[ 8] = (support & nvrhi::FormatSupport::ShaderSample) != 0   ? 'S' : '.';
            features[ 9] = (support & nvrhi::FormatSupport::ShaderUavLoad) != 0  ? 'l' : '.';
            features[10] = (support & nvrhi::FormatSupport::ShaderUavStore) != 0 ? 's' : '.';
            features[11] = (support & nvrhi::FormatSupport::ShaderAtomic) != 0   ? 'A' : '.';
            features[12] = 0;

            Log(Info, "%17s: %s", formatInfo.name, features);
        }
    }

    {
        std::shared_ptr<BasicRenderer> demo = std::make_shared<BasicRenderer>(deviceManager, sceneName);
		std::shared_ptr<UIRenderPass> gui = std::make_shared<UIRenderPass>(deviceManager);

		gui->AddRenderPass(demo);

        deviceManager->AddRenderPassToBack(demo.get());
        deviceManager->AddRenderPassToBack(gui.get());

        deviceManager->RunMessageLoop();
    }

    deviceManager->Shutdown();
    delete deviceManager;

	return 0;
}
