#pragma once

#include <nvrhi/nvrhi.h>
#include <donut/app/DeviceManager.h>

#include "Fluxel.h"

NAMESPACE_BEGIN(fluxel)

class CommonDeviceObject {
public:
	explicit CommonDeviceObject(donut::app::DeviceManager *deviceManager)
		: m_deviceManager(deviceManager) {}

	virtual ~CommonDeviceObject() = default;

	[[nodiscard]] donut::app::DeviceManager *getDeviceManager() const { return m_deviceManager; }
	[[nodiscard]] nvrhi::IDevice *getDevice() const { return m_deviceManager->GetDevice(); }

private:
	donut::app::DeviceManager* m_deviceManager;
};

NAMESPACE_END(fluxel)
