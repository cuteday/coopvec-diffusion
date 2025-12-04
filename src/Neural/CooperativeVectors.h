#pragma once

#include "Fluxel.h"
#include <nvrhi/nvrhi.h>

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

NAMESPACE_END(fluxel)
