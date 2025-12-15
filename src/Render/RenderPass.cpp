#include "RenderPass.h"

#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/core/vfs/VFS.h>

#include "Logger.h"

NAMESPACE_BEGIN(fluxel)

void RenderData::AddResource(const std::string &name, nvrhi::IResource *resource)
{
	m_Resources[name] = resource;
}

nvrhi::IResource *RenderData::GetResource(const std::string &name)
{
	return m_Resources[name];
}

nvrhi::IResource *& RenderData::operator[](const std::string &name)
{
	return m_Resources[name];
}

RenderPass::RenderPass(donut::app::DeviceManager *deviceManager, const std::string &name) :
	donut::app::IRenderPass(deviceManager), m_Name(name) {}


SceneRenderPass::SceneRenderPass(donut::app::DeviceManager* deviceManager, const std::string& name)
    : RenderPass(deviceManager, name)
    , m_SceneLoaded(false)
    , m_AllTexturesFinalized(false)
    , m_IsAsyncLoad(true)
{
}

void SceneRenderPass::Render(nvrhi::IFramebuffer* framebuffer)
{
    if (m_TextureCache)
    {
        bool anyTexturesProcessed = m_TextureCache->ProcessRenderingThreadCommands(*m_CommonPasses, 20.f);

        if (m_SceneLoaded && !anyTexturesProcessed)
            m_AllTexturesFinalized = true;
    }
    else
        m_AllTexturesFinalized = true;

    if (!m_SceneLoaded || !m_AllTexturesFinalized)
    {
        RenderSplashScreen(framebuffer);
        return;
    }

    if (m_SceneLoaded && m_SceneLoadingThread)
    {
        m_SceneLoadingThread->join();
        m_SceneLoadingThread = nullptr;

        // SceneLoaded() would already get called from
        // BeginLoadingScene() in case of synchronous loads
        SceneLoaded();
    }

    RenderScene(framebuffer);
}

void SceneRenderPass::RenderScene(nvrhi::IFramebuffer* framebuffer)
{

}

void SceneRenderPass::RenderSplashScreen(nvrhi::IFramebuffer* framebuffer)
{

}

void SceneRenderPass::SceneUnloading()
{

}

void SceneRenderPass::SceneLoaded()
{
    if (m_TextureCache)
    {
        m_TextureCache->ProcessRenderingThreadCommands(*m_CommonPasses, 0.f);
        m_TextureCache->LoadingFinished();
    }
    m_SceneLoaded = true;
}

void SceneRenderPass::SetAsynchronousLoadingEnabled(bool enabled)
{
    m_IsAsyncLoad = enabled;
}

bool SceneRenderPass::IsSceneLoading() const
{
    return m_SceneLoadingThread != nullptr;
}

bool SceneRenderPass::IsSceneLoaded() const
{
    return m_SceneLoaded;
}

void SceneRenderPass::BeginLoadingScene(std::shared_ptr<donut::vfs::IFileSystem> fs, const std::filesystem::path& sceneFileName)
{
    if (m_SceneLoaded)
        SceneUnloading();

    m_SceneLoaded = false;
    m_AllTexturesFinalized = false;

    if (m_TextureCache)
    {
        m_TextureCache->Reset();
    }
    GetDevice()->waitForIdle();
    GetDevice()->runGarbageCollection();


    if (m_IsAsyncLoad)
    {
        m_SceneLoadingThread = std::make_unique<std::thread>([this, fs, sceneFileName]() {
			m_SceneLoaded = LoadScene(fs, sceneFileName);
			});
    }
    else
    {
        m_SceneLoaded = LoadScene(fs, sceneFileName);
        SceneLoaded();
    }
}

std::shared_ptr<donut::engine::CommonRenderPasses> SceneRenderPass::GetCommonPasses() const
{
    return m_CommonPasses;
}


UIRenderPass::UIRenderPass(donut::app::DeviceManager *deviceManager) :
	ImGui_Renderer(deviceManager) {
	std::filesystem::path frameworkShaderPath = donut::app::GetDirectoryWithExecutable() / "shaders/framework" / donut::app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

	m_rootFS = std::make_shared<donut::vfs::RootFileSystem>();
	m_rootFS->mount("/shaders/donut", frameworkShaderPath);

	m_shaderFactory = std::make_shared<donut::engine::ShaderFactory>(GetDevice(), m_rootFS, "/shaders");

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
				// Get a unique ID for each passes, append the id after "##" to avoid conflicts
				std::string uniqueId = renderPassPtr->GetName() + "##" + std::to_string(std::hash<std::string>{}(renderPassPtr->GetName()));
				if (ImGui::CollapsingHeader(uniqueId.c_str())) {
					bool enabled = renderPassPtr->IsEnabled();
					ImGui::Indent();
					if (ImGui::Checkbox(("Enabled##" + uniqueId).c_str(), &enabled)) {
						renderPassPtr->SetEnabled(enabled);
					}
					if (enabled) {
						renderPassPtr->BuildUI();
					}
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
