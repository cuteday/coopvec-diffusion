#include "RenderPass.h"

#include <donut/engine/ShaderFactory.h>

#include "Logger.h"

NAMESPACE_BEGIN(fluxel)

RenderPass::RenderPass(donut::app::DeviceManager *deviceManager, const std::string& name)
    : donut::app::IRenderPass(deviceManager), m_name(name) {}

RenderPass::~RenderPass() {}

UIRenderPass::UIRenderPass(donut::app::DeviceManager *deviceManager) :
	ImGui_Renderer(deviceManager) {
	std::filesystem::path frameworkShaderPath = donut::app::GetDirectoryWithExecutable() / "shaders/framework" / donut::app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

	m_rootFS = std::make_shared<donut::vfs::RootFileSystem>();
	m_rootFS->mount("/shaders/donut", frameworkShaderPath);

	m_shaderFactory =
		std::make_shared<donut::engine::ShaderFactory>(GetDevice(), m_rootFS, "/shaders");

	if (!Init(m_shaderFactory)) {
		Log(Error, "Failed to initialize ImGui renderer");
		return;
	}
}

UIRenderPass::~UIRenderPass() {}

void UIRenderPass::AddRenderPass(std::weak_ptr<RenderPass> renderPass)
{
	m_renderPasses.push_back(renderPass);
}

void UIRenderPass::Render(nvrhi::IFramebuffer *framebuffer)
{
    if (!imgui_nvrhi)
        return;

	buildUI();

	if (ImGui::CollapsingHeader("Render Passes")) {
		ImGui::Indent();
		for (auto &renderPass : m_renderPasses) {
			if (auto renderPassPtr = renderPass.lock(); renderPassPtr) {
				if (ImGui::CollapsingHeader(renderPassPtr->GetName().c_str())) {
					ImGui::Indent();
					renderPassPtr->BuildUI();
					ImGui::Unindent();
				}
			}
		}
		ImGui::Unindent();
	}

    ImGui::Render();
    imgui_nvrhi->render(framebuffer);
    m_imguiFrameOpened = false;
}

NAMESPACE_END(fluxel)
