#pragma once
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

using namespace fluxel;

class BasicRenderer : public SceneRenderPass {
public:

class RenderTargets : public donut::render::GBufferRenderTargets
{
	public:
		nvrhi::TextureHandle HdrColor;
		nvrhi::TextureHandle LdrColor;
		nvrhi::TextureHandle MaterialIDs;
		nvrhi::TextureHandle ResolvedColor;
		nvrhi::TextureHandle TemporalFeedback1;
		nvrhi::TextureHandle TemporalFeedback2;
		nvrhi::TextureHandle AmbientOcclusion;

		nvrhi::HeapHandle Heap;

		std::shared_ptr<donut::engine::FramebufferFactory> ForwardFramebuffer;
		std::shared_ptr<donut::engine::FramebufferFactory> HdrFramebuffer;
		std::shared_ptr<donut::engine::FramebufferFactory> LdrFramebuffer;
		std::shared_ptr<donut::engine::FramebufferFactory> ResolvedFramebuffer;
		std::shared_ptr<donut::engine::FramebufferFactory> MaterialIDFramebuffer;

		void Init(
			nvrhi::IDevice* device,
			dm::uint2 size,
			dm::uint sampleCount,
			bool enableMotionVectors,
			bool useReverseProjection) override;

		[[nodiscard]] bool IsUpdateRequired(dm::uint2 size, dm::uint sampleCount) const;

		void Clear(nvrhi::ICommandList* commandList) override;
	};

	enum class AntiAliasingMode
	{
		NONE,
		TEMPORAL,
		DLSS,
		MSAA_2X,
		MSAA_4X,
		MSAA_8X
	};

	struct UIData
	{
		bool                                ShowUI = true;
		bool                                ShowConsole = false;
		bool                                UseDeferredShading = true;
		bool                                Stereo = false;
		bool                                EnableSsao = true;
		donut::render::SsaoParameters                      SsaoParams;
		donut::render::ToneMappingParameters               ToneMappingParams;
		donut::render::TemporalAntiAliasingParameters      TemporalAntiAliasingParams;
		donut::render::SkyParameters                       SkyParams;
		enum AntiAliasingMode               AntiAliasingMode = AntiAliasingMode::TEMPORAL;
		donut::render::TemporalAntiAliasingJitter     TemporalAntiAliasingJitter = donut::render::TemporalAntiAliasingJitter::MSAA;
		bool                                EnableVsync = true;
		bool                                ShaderReoladRequested = false;
		bool                                EnableProceduralSky = true;
		bool                                EnableBloom = true;
		bool                                DlssAvailable = false;
		float                               BloomSigma = 32.f;
		float                               BloomAlpha = 0.05f;
		bool                                EnableTranslucency = true;
		bool                                EnableMaterialEvents = false;
		bool                                EnableShadows = true;
		float                               AmbientIntensity = 1.0f;
		bool                                EnableLightProbe = true;
		float                               LightProbeDiffuseScale = 1.f;
		float                               LightProbeSpecularScale = 1.f;
		float                               CsmExponent = 4.f;
		bool                                DisplayShadowMap = false;
		bool                                UseThirdPersonCamera = false;
		bool                                EnableAnimations = false;
		bool                                TestMipMapGen = false;
		std::shared_ptr<donut::engine::Material>           SelectedMaterial;
		std::shared_ptr<donut::engine::SceneGraphNode>     SelectedNode;
		std::string                         ScreenshotFileName;
		std::shared_ptr<donut::engine::SceneCamera>        ActiveSceneCamera;
	};

protected:
    typedef SceneRenderPass Super;

    std::shared_ptr<donut::vfs::RootFileSystem>     m_RootFs;
    std::shared_ptr<donut::vfs::NativeFileSystem>   m_NativeFs;
	std::vector<std::string>            m_SceneFilesAvailable;
    std::string                         m_CurrentSceneName;
    std::filesystem::path               m_SceneDir;
	std::shared_ptr<donut::engine::Scene>				m_Scene;
	std::shared_ptr<donut::engine::ShaderFactory>      m_ShaderFactory;
    std::shared_ptr<donut::engine::DirectionalLight>   m_SunLight;
    std::shared_ptr<donut::render::CascadedShadowMap>  m_ShadowMap;
    std::shared_ptr<donut::engine::FramebufferFactory> m_ShadowFramebuffer;
    std::shared_ptr<donut::render::DepthPass>          m_ShadowDepthPass;
    std::shared_ptr<donut::render::InstancedOpaqueDrawStrategy> m_OpaqueDrawStrategy;
    std::shared_ptr<donut::render::TransparentDrawStrategy> m_TransparentDrawStrategy;
    std::unique_ptr<RenderTargets>      m_RenderTargets;
    std::shared_ptr<donut::render::ForwardShadingPass> m_ForwardPass;
    std::unique_ptr<donut::render::GBufferFillPass>    m_GBufferPass;
    std::unique_ptr<donut::render::DeferredLightingPass> m_DeferredLightingPass;
    std::unique_ptr<donut::render::SkyPass>            m_SkyPass;
    std::unique_ptr<donut::render::TemporalAntiAliasingPass> m_TemporalAntiAliasingPass;
#if DONUT_WITH_DLSS
    std::unique_ptr<DLSS>               m_DLSS;
#endif
    std::unique_ptr<donut::render::BloomPass>          m_BloomPass;
    std::unique_ptr<donut::render::ToneMappingPass>    m_ToneMappingPass;
    std::unique_ptr<donut::render::SsaoPass>           m_SsaoPass;
    std::shared_ptr<donut::render::LightProbeProcessingPass> m_LightProbePass;
    std::unique_ptr<donut::render::MaterialIDPass>     m_MaterialIDPass;
    std::unique_ptr<donut::render::PixelReadbackPass>  m_PixelReadbackPass;
    std::unique_ptr<donut::render::MipMapGenPass>      m_MipMapGenPass;

    std::shared_ptr<donut::engine::IView>              m_View;
    std::shared_ptr<donut::engine::IView>              m_ViewPrevious;

    nvrhi::CommandListHandle            m_CommandList;
    bool                                m_PreviousViewsValid = false;
    donut::app::FirstPersonCamera                   m_FirstPersonCamera;
    donut::app::ThirdPersonCamera                   m_ThirdPersonCamera;
    donut::engine::BindingCache                        m_BindingCache;

    float                               m_CameraVerticalFov = 60.f;
    dm::float3                              m_AmbientTop = 0.f;
    dm::float3                              m_AmbientBottom = 0.f;
    dm::uint2                               m_PickPosition = 0u;
    bool                                m_Pick = false;

    std::vector<std::shared_ptr<donut::engine::LightProbe>> m_LightProbes;
    nvrhi::TextureHandle                m_LightProbeDiffuseTexture;
    nvrhi::TextureHandle                m_LightProbeSpecularTexture;

    float                               m_WallclockTime = 0.f;

	// UI related data
    UIData                             m_ui = {};

	std::unique_ptr<donut::app::ImGui_Console> m_console;
	std::shared_ptr<donut::engine::Light> m_SelectedLight;

public:

    BasicRenderer(donut::app::DeviceManager* deviceManager, const std::string& sceneName);

	std::shared_ptr<donut::vfs::IFileSystem> GetRootFs() const;

    donut::app::BaseCamera& GetActiveCamera() const;

	std::vector<std::string> const& GetAvailableScenes() const;

    std::filesystem::path const& GetSceneDir() const;

    std::string GetCurrentSceneName() const;

    void SetCurrentSceneName(const std::string& sceneName);

    void CopyActiveCameraToFirstPerson();

    virtual bool KeyboardUpdate(int key, int scancode, int action, int mods) override;

    virtual bool MousePosUpdate(double xpos, double ypos) override;

    virtual bool MouseButtonUpdate(int button, int action, int mods) override;

    virtual bool MouseScrollUpdate(double xoffset, double yoffset) override;

    virtual void Animate(float fElapsedTimeSeconds) override;

    virtual void SceneUnloading() override;

    virtual bool LoadScene(std::shared_ptr<donut::vfs::IFileSystem> fs, const std::filesystem::path& fileName) override;

    virtual void SceneLoaded() override;

    void PointThirdPersonCameraAt(const std::shared_ptr<donut::engine::SceneGraphNode>& node);

    bool IsStereo();

    std::shared_ptr<donut::engine::TextureCache> GetTextureCache();

    std::shared_ptr<donut::engine::Scene> GetScene();

    bool SetupView();

    void CreateRenderPasses(bool& exposureResetRequired);

    virtual void RenderSplashScreen(nvrhi::IFramebuffer* framebuffer) override;

    virtual void RenderScene(nvrhi::IFramebuffer* framebuffer) override;

    std::shared_ptr<donut::engine::ShaderFactory> GetShaderFactory();

    std::vector<std::shared_ptr<donut::engine::LightProbe>>& GetLightProbes();

    void CreateLightProbes(uint32_t numProbes);

    void RenderLightProbe(donut::engine::LightProbe& probe);

	void BuildUI() override;
};
