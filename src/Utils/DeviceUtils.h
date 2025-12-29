#pragma once

#include <nvrhi/nvrhi.h>

#include "Fluxel.h"

namespace donut::app
{
struct DeviceCreationParameters;
class DeviceManager;
} // namespace donut::app

NAMESPACE_BEGIN(fluxel)
// Select a preferred adapter: prefer NVIDIA (0x10DE) with max VRAM, else overall max VRAM.
// Returns true if an adapter was selected and writes deviceParams.adapterIndex.
bool SelectPreferredAdapter(donut::app::DeviceManager* deviceManager, donut::app::DeviceCreationParameters& deviceParams);

NAMESPACE_END(fluxel)
