#pragma once

#include <nvrhi/nvrhi.h>
#include <donut/app/DeviceManager.h>

#include "Fluxel.h"

NAMESPACE_BEGIN(fluxel)

class CommonDeviceObject {
public:
	explicit CommonDeviceObject(nvrhi::IDevice *device)
		: m_device(device) {}

	virtual ~CommonDeviceObject() = default;

	[[nodiscard]] nvrhi::IDevice *getDevice() const { return m_device; }

private:
	nvrhi::IDevice* m_device;
};

NAMESPACE_END(fluxel)
