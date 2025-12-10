#pragma once

#include <memory>
#include <nvrhi/nvrhi.h>

#include <donut/app/DeviceManager.h>
#include <donut/app/ApplicationBase.h>
#include <donut/app/imgui_renderer.h>

#include <Fluxel.h>

NAMESPACE_BEGIN(fluxel)

class RenderPass : public donut::app::IRenderPass{
public:
    RenderPass(donut::app::DeviceManager *deviceManager, const std::string& name = "");
	~RenderPass();

	const std::string &GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	virtual void BuildUI() {}

protected:
	std::string m_name;
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
