#if DONUT_WITH_DX12
#include "../../external/dx12-agility-sdk/build/native/include/d3d12.h"
#endif

#include "DeviceUtils.h"

#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <vector>
#include <cstdint>

#if DONUT_WITH_DX12
extern "C"
{
    _declspec(dllexport) extern const uint32_t D3D12SDKVersion = DONUT_D3D_AGILITY_PREVIEW_SDK_VERSION;
    _declspec(dllexport) extern const char* D3D12SDKPath = ".\\d3d12\\";
}

static bool g_dx12DeveloperModeEnabled = false;
#endif

#if DONUT_WITH_VULKAN
static bool g_vulkanStorageBuffer16BitAccess = false;
static bool g_vulkanFloat16Supported = false;
#endif

// For the purposes of the SDK, when using Vulkan VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME is set as a required device extension.
void SetCoopVectorExtensionParameters(donut::app::DeviceCreationParameters& deviceParams, nvrhi::GraphicsAPI graphicsApi, bool enableSharedMemory, char const* windowTitle)
{
#if DONUT_WITH_VULKAN
    if (graphicsApi == nvrhi::GraphicsAPI::VULKAN)
    {
        if (enableSharedMemory)
        {
#ifdef _WIN32
            deviceParams.requiredVulkanDeviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#else
            deviceParams.requiredVulkanDeviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
        }

        deviceParams.requiredVulkanDeviceExtensions.push_back(VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME);
        deviceParams.requiredVulkanDeviceExtensions.push_back(VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME);

        // vkCmdCopyImage: Dest image pRegion[0] x-dimension offset [0] + extent [4] exceeds subResource width [2]
        // vkCmdCopyImage: Dest image pRegion[0] y-dimension offset [0] + extent [4] exceeds subResource height [2]
        // These errors happen during copies from block textures to BCn textures at the last 2 mips, no way around it.
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x38b5face);
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x4bb17a0e);

        // The following warnings are related to the Cooperative Vector extension that the validation layers don't know.
        // SPIR-V module not valid: Invalid capability operand: 5394
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x6bbb14);

        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0xa5625282);
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x79de34d4);
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x901f59ec);

        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x605314fa);
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x938b32);

        // vkCreateShaderModule(): A SPIR-V Capability (Unhandled OpCapability) was declared that is not supported by Vulkan.
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x2c00a3d6);
        // A SPIR-V Extension (SPV_NV_cooperative_vector) was declared that is not supported by Vulkan.
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0xffffffffd80a42ae);

        // fragment shader writes to output location 1 with no matching attachment
        // This happens in the forward shading pass for transmissive materials. Difficult to work around.
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x609a13b);

        // Add feature structures querying for cooperative vector support and DP4a support
        static VkPhysicalDeviceCooperativeVectorFeaturesNV cooperativeVectorFeatures{};
        cooperativeVectorFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV;

        static VkPhysicalDeviceShaderReplicatedCompositesFeaturesEXT shaderReplicatedCompositesFeatures{};
        shaderReplicatedCompositesFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_REPLICATED_COMPOSITES_FEATURES_EXT;
        shaderReplicatedCompositesFeatures.pNext = &cooperativeVectorFeatures;

        static VkPhysicalDeviceVulkan11Features vulkan11Features{};
        vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        vulkan11Features.pNext = &shaderReplicatedCompositesFeatures;

        static VkPhysicalDeviceVulkan12Features vulkan12Features{};
        vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan12Features.pNext = &vulkan11Features;
        deviceParams.physicalDeviceFeatures2Extensions = &vulkan12Features;

        // Set the callback to modify some bits in VkDeviceCreateInfo before creating the device
        deviceParams.deviceCreateInfoCallback = [](VkDeviceCreateInfo& info) {
            const_cast<VkPhysicalDeviceFeatures*>(info.pEnabledFeatures)->shaderInt16 = true;
            const_cast<VkPhysicalDeviceFeatures*>(info.pEnabledFeatures)->fragmentStoresAndAtomics = true;
            const_cast<VkPhysicalDeviceFeatures*>(info.pEnabledFeatures)->shaderInt64 = true;

            // Iterate through the structure chain and find the structures to patch
            VkBaseOutStructure* pCurrent = reinterpret_cast<VkBaseOutStructure*>(&info);
            VkBaseOutStructure* pLast = nullptr;
            while (pCurrent)
            {
                if (pCurrent->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES)
                {
                    g_vulkanStorageBuffer16BitAccess = vulkan11Features.storageBuffer16BitAccess;
                    reinterpret_cast<VkPhysicalDeviceVulkan11Features*>(pCurrent)->storageBuffer16BitAccess = g_vulkanStorageBuffer16BitAccess;
                }

                if (pCurrent->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
                {
                    g_vulkanFloat16Supported = vulkan12Features.shaderFloat16;
                    reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(pCurrent)->shaderFloat16 = g_vulkanFloat16Supported;
                    reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(pCurrent)->vulkanMemoryModel = true;
                    reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(pCurrent)->vulkanMemoryModelDeviceScope = true;
                }

                pLast = pCurrent;
                pCurrent = pCurrent->pNext;
            }

            if (pLast && shaderReplicatedCompositesFeatures.shaderReplicatedComposites)
            {
                pLast->pNext = reinterpret_cast<VkBaseOutStructure*>(&shaderReplicatedCompositesFeatures);
                shaderReplicatedCompositesFeatures.pNext = nullptr;
                pLast = pLast->pNext;
            }

            // If cooperative vector is supported, add a feature structure enabling it on the device
            if (pLast && cooperativeVectorFeatures.cooperativeVector)
            {
                pLast->pNext = reinterpret_cast<VkBaseOutStructure*>(&cooperativeVectorFeatures);
                cooperativeVectorFeatures.pNext = nullptr;
            }
        };
    }
#endif

#if DONUT_WITH_DX12
    if (graphicsApi == nvrhi::GraphicsAPI::D3D12)
    {
        UUID Features[] = { D3D12ExperimentalShaderModels, D3D12CooperativeVectorExperiment };
        HRESULT hr = D3D12EnableExperimentalFeatures(_countof(Features), Features, nullptr, nullptr);

        if (FAILED(hr))
        {
            char const* messageText =
                "Couldn't enable D3D12 experimental shader models. Cooperative Vector features will not be available.\n"
                "Please make sure that Developer Mode is enabled in the Windows system settings.";

            if (windowTitle)
            {
                MessageBoxA(NULL, messageText, windowTitle, MB_ICONWARNING);
            }
            else
            {
                donut::log::warning("%s", messageText);
            }
        }
        else
        {
            g_dx12DeveloperModeEnabled = true;
        }
    }
#endif
}

// Call after device creation to verify the extension has been enabled
bool CoopVectorExtensionSupported(donut::app::DeviceManager* deviceManager)
{
    std::vector<std::string> extensions;

    deviceManager->GetEnabledVulkanDeviceExtensions(extensions);
    for (std::string extension : extensions)
    {
        if (extension == "VK_NV_cooperative_vector")
        {
            return true;
        }
    }
    return false;
}

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
