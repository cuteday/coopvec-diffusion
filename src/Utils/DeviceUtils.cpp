#if DONUT_WITH_DX12
#include "../../external/dx12-agility-sdk/build/native/include/d3d12.h"
#endif

#include "DeviceUtils.h"

#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <vector>
#include <cstdint>

NAMESPACE_BEGIN(fluxel)

bool SelectPreferredAdapter(donut::app::DeviceManager* deviceManager, donut::app::DeviceCreationParameters& deviceParams)
{
    if (!deviceManager)
        return false;

    std::vector<donut::app::AdapterInfo> adapters;
    if (!deviceManager->EnumerateAdapters(adapters) || adapters.empty())
        return false;

    int selectedIndex = -1;
    uint64_t maxVram = 0;

    // Prefer NVIDIA (0x10DE) adapters with the most VRAM.
    for (size_t i = 0; i < adapters.size(); ++i)
    {
        if (adapters[i].vendorID == 0x10DE)
        {
            if (adapters[i].dedicatedVideoMemory > maxVram)
            {
                maxVram = adapters[i].dedicatedVideoMemory;
                selectedIndex = int(i);
            }
        }
    }

    // Fallback: adapter with the most VRAM overall.
    if (selectedIndex < 0)
    {
        maxVram = 0;
        for (size_t i = 0; i < adapters.size(); ++i)
        {
            if (adapters[i].dedicatedVideoMemory > maxVram)
            {
                maxVram = adapters[i].dedicatedVideoMemory;
                selectedIndex = int(i);
            }
        }
    }

    if (selectedIndex >= 0)
    {
        deviceParams.adapterIndex = selectedIndex;
        return true;
    }

    return false;
}

NAMESPACE_END(fluxel)
