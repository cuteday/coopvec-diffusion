#pragma once

#include <memory>
#include <nvrhi/nvrhi.h>

#include <donut/app/DeviceManager.h>
#include <donut/app/ApplicationBase.h>
#include <donut/app/imgui_renderer.h>

#include <Fluxel.h>
#include <Object.h>

NAMESPACE_BEGIN(fluxel)

class RenderData : public CommonDeviceObject {
public:
	explicit RenderData(nvrhi::IDevice *device)
		: CommonDeviceObject(device) {}

	virtual ~RenderData() = default;

	void AddResource(const std::string &name, nvrhi::IResource *resource);
	nvrhi::IResource *GetResource(const std::string &name);

	nvrhi::IResource *& operator[](const std::string &name);

private:
	std::unordered_map<std::string, nvrhi::IResource*> m_Resources;
};

class RenderPass : public donut::app::IRenderPass{
public:
    RenderPass(donut::app::DeviceManager *deviceManager, const std::string& name = "");
	virtual ~RenderPass() = default;

	virtual const std::string &GetName() const { return m_Name; }
	virtual void SetName(const std::string& name) { m_Name = name; }
	virtual bool IsEnabled() const { return m_Enabled; }
	virtual void SetEnabled(bool enabled) { m_Enabled = enabled; }

	virtual void Render(nvrhi::IFramebuffer *framebuffer) override {}
	virtual void Render(nvrhi::ICommandList *commandList, std::shared_ptr<RenderData> renderData) {}

	virtual void BuildUI() {}

protected:
	std::string m_Name;
	bool m_Enabled = true;
};

class SceneRenderPass : public RenderPass {
private:
	bool m_SceneLoaded;
	bool m_AllTexturesFinalized;

protected:
	typedef IRenderPass Super;

	std::shared_ptr<donut::engine::TextureCache> m_TextureCache;
	std::unique_ptr<std::thread> m_SceneLoadingThread;
	std::shared_ptr<donut::engine::CommonRenderPasses> m_CommonPasses;

	bool m_IsAsyncLoad;

public:
	SceneRenderPass(donut::app::DeviceManager* deviceManager, const std::string& name = "");
	virtual ~SceneRenderPass() = default;

	virtual void SetLatewarpOptions() override { }
	virtual void Render(nvrhi::IFramebuffer* framebuffer) override;

	virtual void RenderScene(nvrhi::IFramebuffer* framebuffer);
	virtual void RenderSplashScreen(nvrhi::IFramebuffer* framebuffer);
	virtual void BeginLoadingScene(std::shared_ptr<donut::vfs::IFileSystem> fs, const std::filesystem::path& sceneFileName);
	virtual bool LoadScene(std::shared_ptr<donut::vfs::IFileSystem> fs, const std::filesystem::path& sceneFileName) = 0;
	virtual void SceneUnloading();
	virtual void SceneLoaded();

	void SetAsynchronousLoadingEnabled(bool enabled);
	bool IsSceneLoading() const;
	bool IsSceneLoaded() const;

	virtual void BuildUI() override {}

	std::shared_ptr<donut::engine::CommonRenderPasses> GetCommonPasses() const;
};

class UIRenderPass : public donut::app::ImGui_Renderer {
public:
    UIRenderPass(donut::app::DeviceManager *deviceManager);
	~UIRenderPass();

	void Render(nvrhi::IFramebuffer *framebuffer) override;
	void AddRenderPass(std::weak_ptr<RenderPass> renderPass);

protected:
	virtual void buildUI(void) override {}

	std::list<std::weak_ptr<RenderPass>> m_renderPasses;

	std::shared_ptr<donut::vfs::RootFileSystem> m_rootFS;
	std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;
};

NAMESPACE_END(fluxel)
