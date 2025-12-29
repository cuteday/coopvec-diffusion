#pragma once

#include "Fluxel.h"
#include <nvrhi/nvrhi.h>

namespace donut::app
{
struct DeviceCreationParameters;
class DeviceManager;
} // namespace donut::app

NAMESPACE_BEGIN(fluxel)

struct CoopVectorFeatures
{
    bool inferenceSupported = false;
    bool trainingSupported = false;
    bool fp16InferencingSupported = false;
    bool fp16TrainingSupported = false;
};

class GraphicsResources
{
public:
    GraphicsResources(nvrhi::DeviceHandle device);
    ~GraphicsResources();
    CoopVectorFeatures GetCoopVectorFeatures() const
    {
        return m_coopVectorFeatures;
    }

private:
    CoopVectorFeatures m_coopVectorFeatures;
};

void SetCoopVectorExtensionParameters(donut::app::DeviceCreationParameters& deviceParams, nvrhi::GraphicsAPI graphicsApi, bool enableSharedMemory, char const* windowTitle);

// Call after device creation to verify the extension has been enabled
bool CoopVectorExtensionSupported(donut::app::DeviceManager *deviceManager);

NAMESPACE_END(fluxel)
