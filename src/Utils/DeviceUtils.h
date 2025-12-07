#pragma once

#include <nvrhi/nvrhi.h>

namespace donut::app
{
struct DeviceCreationParameters;
class DeviceManager;
} // namespace donut::app

void SetCoopVectorExtensionParameters(donut::app::DeviceCreationParameters& deviceParams, nvrhi::GraphicsAPI graphicsApi, bool enableSharedMemory, char const* windowTitle);

// Call after device creation to verify the extension has been enabled
bool CoopVectorExtensionSupported(donut::app::DeviceManager *deviceManager);

// Select a preferred adapter: prefer NVIDIA (0x10DE) with max VRAM, else overall max VRAM.
// Returns true if an adapter was selected and writes deviceParams.adapterIndex.
bool SelectPreferredAdapter(donut::app::DeviceManager* deviceManager, donut::app::DeviceCreationParameters& deviceParams);

