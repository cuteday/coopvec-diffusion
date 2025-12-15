#include "Renderer.h"

// BasicRenderer::RenderTargets member functions

void BasicRenderer::RenderTargets::Init(
    nvrhi::IDevice* device,
    dm::uint2 size,
    dm::uint sampleCount,
    bool enableMotionVectors,
    bool useReverseProjection)
{
    GBufferRenderTargets::Init(device, size, sampleCount, enableMotionVectors, useReverseProjection);

    nvrhi::TextureDesc desc;
    desc.width = size.x;
    desc.height = size.y;
    desc.isRenderTarget = true;
    desc.useClearValue = true;
    desc.clearValue = nvrhi::Color(1.f);
    desc.sampleCount = sampleCount;
    desc.dimension = sampleCount > 1 ? nvrhi::TextureDimension::Texture2DMS : nvrhi::TextureDimension::Texture2D;
    desc.keepInitialState = true;
    desc.isVirtual = device->queryFeatureSupport(nvrhi::Feature::VirtualResources);

    desc.clearValue = nvrhi::Color(0.f);
    desc.isTypeless = false;
    desc.isUAV = sampleCount == 1;
    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.initialState = nvrhi::ResourceStates::RenderTarget;
    desc.debugName = "HdrColor";
    HdrColor = device->createTexture(desc);

    desc.format = nvrhi::Format::RG16_UINT;
    desc.isUAV = false;
    desc.debugName = "MaterialIDs";
    MaterialIDs = device->createTexture(desc);

    // The render targets below this point are non-MSAA
    desc.sampleCount = 1;
    desc.dimension = nvrhi::TextureDimension::Texture2D;

    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.isUAV = true;
    desc.mipLevels = uint32_t(floorf(::log2f(float(std::max(desc.width, desc.height)))) + 1.f); // Used to test the MipMapGen pass
    desc.debugName = "ResolvedColor";
    ResolvedColor = device->createTexture(desc);

    desc.format = nvrhi::Format::RGBA16_SNORM;
    desc.mipLevels = 1;
    desc.debugName = "TemporalFeedback1";
    TemporalFeedback1 = device->createTexture(desc);
    desc.debugName = "TemporalFeedback2";
    TemporalFeedback2 = device->createTexture(desc);

    desc.format = nvrhi::Format::SRGBA8_UNORM;
    desc.isUAV = false;
    desc.debugName = "LdrColor";
    LdrColor = device->createTexture(desc);

    desc.format = nvrhi::Format::R8_UNORM;
    desc.isUAV = true;
    desc.debugName = "AmbientOcclusion";
    AmbientOcclusion = device->createTexture(desc);

    if (desc.isVirtual)
    {
        uint64_t heapSize = 0;
        nvrhi::ITexture* const textures[] = {
            HdrColor,
            MaterialIDs,
            ResolvedColor,
            TemporalFeedback1,
            TemporalFeedback2,
            LdrColor,
            AmbientOcclusion
        };

        for (auto texture : textures)
        {
            nvrhi::MemoryRequirements memReq = device->getTextureMemoryRequirements(texture);
            heapSize = nvrhi::align(heapSize, memReq.alignment);
            heapSize += memReq.size;
        }

        nvrhi::HeapDesc heapDesc;
        heapDesc.type = nvrhi::HeapType::DeviceLocal;
        heapDesc.capacity = heapSize;
        heapDesc.debugName = "RenderTargetHeap";

        Heap = device->createHeap(heapDesc);

        uint64_t offset = 0;
        for (auto texture : textures)
        {
            nvrhi::MemoryRequirements memReq = device->getTextureMemoryRequirements(texture);
            offset = nvrhi::align(offset, memReq.alignment);

            device->bindTextureMemory(texture, Heap, offset);

            offset += memReq.size;
        }
    }

    ForwardFramebuffer = std::make_shared<donut::engine::FramebufferFactory>(device);
    ForwardFramebuffer->RenderTargets = { HdrColor };
    ForwardFramebuffer->DepthTarget = Depth;

    HdrFramebuffer = std::make_shared<donut::engine::FramebufferFactory>(device);
    HdrFramebuffer->RenderTargets = { HdrColor };

    LdrFramebuffer = std::make_shared<donut::engine::FramebufferFactory>(device);
    LdrFramebuffer->RenderTargets = { LdrColor };

    ResolvedFramebuffer = std::make_shared<donut::engine::FramebufferFactory>(device);
    ResolvedFramebuffer->RenderTargets = { ResolvedColor };

    MaterialIDFramebuffer = std::make_shared<donut::engine::FramebufferFactory>(device);
    MaterialIDFramebuffer->RenderTargets = { MaterialIDs };
    MaterialIDFramebuffer->DepthTarget = Depth;
}

bool BasicRenderer::RenderTargets::IsUpdateRequired(dm::uint2 size, dm::uint sampleCount) const
{
    if (any(m_Size != size) || m_SampleCount != sampleCount)
        return true;

    return false;
}

void BasicRenderer::RenderTargets::Clear(nvrhi::ICommandList* commandList)
{
    GBufferRenderTargets::Clear(commandList);

    commandList->clearTextureFloat(HdrColor, nvrhi::AllSubresources, nvrhi::Color(0.f));
    commandList->clearTextureFloat(LdrColor, nvrhi::AllSubresources, nvrhi::Color(0.f));
    commandList->clearTextureFloat(ResolvedColor, nvrhi::AllSubresources, nvrhi::Color(0.f));
}

// BasicRenderer member functions

BasicRenderer::BasicRenderer(donut::app::DeviceManager* deviceManager, const std::string& sceneName)
    : Super(deviceManager, "Simple Renderer")
    , m_BindingCache(deviceManager->GetDevice())
{
    m_RootFs = std::make_shared<donut::vfs::RootFileSystem>();

    std::filesystem::path mediaDir = donut::app::GetDirectoryWithExecutable().parent_path().parent_path() / "assets";
    std::filesystem::path frameworkShaderDir = donut::app::GetDirectoryWithExecutable() / "shaders/framework" / donut::app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

    m_RootFs->mount("/media", mediaDir);
    m_RootFs->mount("/shaders/donut", frameworkShaderDir);

    m_NativeFs = std::make_shared<donut::vfs::NativeFileSystem>();

    m_SceneDir = mediaDir / "scenes";
    m_SceneFilesAvailable = donut::app::FindScenes(*m_NativeFs, m_SceneDir);

    if (sceneName.empty() && m_SceneFilesAvailable.empty())
    {
        Log(Fatal, "No scene file found in media folder '%s'\n"
            "Please make sure that folder contains valid scene files.", m_SceneDir.generic_string().c_str());
    }

    m_TextureCache = std::make_shared<donut::engine::TextureCache>(GetDevice(), m_NativeFs, nullptr);

    m_ShaderFactory = std::make_shared<donut::engine::ShaderFactory>(GetDevice(), m_RootFs, "/shaders");
    m_CommonPasses = std::make_shared<donut::engine::CommonRenderPasses>(GetDevice(), m_ShaderFactory);

    m_OpaqueDrawStrategy = std::make_shared<donut::render::InstancedOpaqueDrawStrategy>();
    m_TransparentDrawStrategy = std::make_shared<donut::render::TransparentDrawStrategy>();


    const nvrhi::Format shadowMapFormats[] = {
        nvrhi::Format::D24S8,
        nvrhi::Format::D32,
        nvrhi::Format::D16,
        nvrhi::Format::D32S8 };

    const nvrhi::FormatSupport shadowMapFeatures =
        nvrhi::FormatSupport::Texture |
        nvrhi::FormatSupport::DepthStencil |
        nvrhi::FormatSupport::ShaderLoad;

    nvrhi::Format shadowMapFormat = nvrhi::utils::ChooseFormat(GetDevice(), shadowMapFeatures, shadowMapFormats, std::size(shadowMapFormats));

    m_ShadowMap = std::make_shared<donut::render::CascadedShadowMap>(GetDevice(), 2048, 4, 0, shadowMapFormat);
    m_ShadowMap->SetupProxyViews();

    m_ShadowFramebuffer = std::make_shared<donut::engine::FramebufferFactory>(GetDevice());
    m_ShadowFramebuffer->DepthTarget = m_ShadowMap->GetTexture();

    donut::render::DepthPass::CreateParameters shadowDepthParams;
    shadowDepthParams.slopeScaledDepthBias = 4.f;
    shadowDepthParams.depthBias = 100;
    m_ShadowDepthPass = std::make_shared<donut::render::DepthPass>(GetDevice(), m_CommonPasses);
    m_ShadowDepthPass->Init(*m_ShaderFactory, shadowDepthParams);

    m_CommandList = GetDevice()->createCommandList();

    m_FirstPersonCamera.SetMoveSpeed(3.0f);
    m_ThirdPersonCamera.SetMoveSpeed(3.0f);

#if DONUT_WITH_DLSS
    // DLSS doesn't need to be re-created when we reload shaders, so we create it here and not in CreateRenderPasses()
    m_DLSS = DLSS::Create(GetDevice(), *m_ShaderFactory, app::GetDirectoryWithExecutable().generic_string());
#endif

    SetAsynchronousLoadingEnabled(true);

    if (sceneName.empty())
        SetCurrentSceneName(donut::app::FindPreferredScene(m_SceneFilesAvailable, "Sponza.gltf"));
    else
        SetCurrentSceneName(sceneName);

    CreateLightProbes(4);

    auto interpreter = std::make_shared<donut::engine::console::Interpreter>();
    // m_console = std::make_unique<ImGui_Console>(interpreter,opts);
}

std::shared_ptr<donut::vfs::IFileSystem> BasicRenderer::GetRootFs() const
{
    return m_RootFs;
}

donut::app::BaseCamera& BasicRenderer::GetActiveCamera() const
{
    return m_ui.UseThirdPersonCamera ? (donut::app::BaseCamera&)m_ThirdPersonCamera : (donut::app::BaseCamera&)m_FirstPersonCamera;
}

std::vector<std::string> const& BasicRenderer::GetAvailableScenes() const
{
    return m_SceneFilesAvailable;
}

std::filesystem::path const& BasicRenderer::GetSceneDir() const
{
    return m_SceneDir;
}

std::string BasicRenderer::GetCurrentSceneName() const
{
    return m_CurrentSceneName;
}

void BasicRenderer::SetCurrentSceneName(const std::string& sceneName)
{
    if (m_CurrentSceneName == sceneName)
        return;

    m_CurrentSceneName = sceneName;

    BeginLoadingScene(m_NativeFs, m_CurrentSceneName);
}

void BasicRenderer::CopyActiveCameraToFirstPerson()
{
    if (m_ui.ActiveSceneCamera)
    {
        dm::affine3 viewToWorld = m_ui.ActiveSceneCamera->GetViewToWorldMatrix();
        dm::float3 cameraPos = viewToWorld.m_translation;
        m_FirstPersonCamera.LookAt(cameraPos, cameraPos + viewToWorld.m_linear.row2, viewToWorld.m_linear.row1);
    }
    else if (m_ui.UseThirdPersonCamera)
    {
        m_FirstPersonCamera.LookAt(m_ThirdPersonCamera.GetPosition(), m_ThirdPersonCamera.GetPosition() + m_ThirdPersonCamera.GetDir(), m_ThirdPersonCamera.GetUp());
    }
}

bool BasicRenderer::KeyboardUpdate(int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        m_ui.ShowUI = !m_ui.ShowUI;
        return true;
    }

    if (key == GLFW_KEY_GRAVE_ACCENT && action == GLFW_PRESS)
    {
        m_ui.ShowConsole = !m_ui.ShowConsole;
        return true;
    }

    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        m_ui.EnableAnimations = !m_ui.EnableAnimations;
        return true;
    }

    if (key == GLFW_KEY_T && action == GLFW_PRESS)
    {
        CopyActiveCameraToFirstPerson();
        if (m_ui.ActiveSceneCamera)
        {
            m_ui.UseThirdPersonCamera = false;
            m_ui.ActiveSceneCamera = nullptr;
        }
        else
        {
            m_ui.UseThirdPersonCamera = !m_ui.UseThirdPersonCamera;
        }
        return true;
    }

    if (!m_ui.ActiveSceneCamera)
        GetActiveCamera().KeyboardUpdate(key, scancode, action, mods);
    return true;
}

bool BasicRenderer::MousePosUpdate(double xpos, double ypos)
{
    if (!m_ui.ActiveSceneCamera)
        GetActiveCamera().MousePosUpdate(xpos, ypos);

    m_PickPosition = dm::uint2(static_cast<dm::uint>(xpos), static_cast<dm::uint>(ypos));

    return true;
}

bool BasicRenderer::MouseButtonUpdate(int button, int action, int mods)
{
    if (!m_ui.ActiveSceneCamera)
        GetActiveCamera().MouseButtonUpdate(button, action, mods);

    if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_2)
        m_Pick = true;

    return true;
}

bool BasicRenderer::MouseScrollUpdate(double xoffset, double yoffset)
{
    if (!m_ui.ActiveSceneCamera)
        GetActiveCamera().MouseScrollUpdate(xoffset, yoffset);

    return true;
}

void BasicRenderer::Animate(float fElapsedTimeSeconds)
{
    if (!m_ui.ActiveSceneCamera)
        GetActiveCamera().Animate(fElapsedTimeSeconds);

    if(m_ToneMappingPass)
        m_ToneMappingPass->AdvanceFrame(fElapsedTimeSeconds);

    if (IsSceneLoaded() && m_ui.EnableAnimations)
    {
        m_WallclockTime += fElapsedTimeSeconds;

        for (const auto& anim : m_Scene->GetSceneGraph()->GetAnimations())
        {
            float duration = anim->GetDuration();
            float integral;
            float animationTime = std::modf(m_WallclockTime / duration, &integral) * duration;
            (void)anim->Apply(animationTime);
        }
    }
}

void BasicRenderer::SceneUnloading()
{
    if (m_ForwardPass) m_ForwardPass->ResetBindingCache();
    if (m_DeferredLightingPass) m_DeferredLightingPass->ResetBindingCache();
    if (m_GBufferPass) m_GBufferPass->ResetBindingCache();
    if (m_LightProbePass) m_LightProbePass->ResetCaches();
    if (m_ShadowDepthPass) m_ShadowDepthPass->ResetBindingCache();
    m_BindingCache.Clear();
    m_SunLight.reset();
    m_ui.SelectedMaterial = nullptr;
    m_ui.SelectedNode = nullptr;

    for (auto probe : m_LightProbes)
    {
        probe->enabled = false;
    }
}

bool BasicRenderer::LoadScene(std::shared_ptr<donut::vfs::IFileSystem> fs, const std::filesystem::path& fileName)
{
    using namespace std::chrono;

    std::unique_ptr<donut::engine::Scene> scene = std::make_unique<donut::engine::Scene>(GetDevice(),
        *m_ShaderFactory, fs, m_TextureCache, nullptr, nullptr);

    auto startTime = high_resolution_clock::now();

    if (scene->Load(fileName))
    {
        m_Scene = std::move(scene);

        auto endTime = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(endTime - startTime).count();
        Log(Info, "Scene loading time: %llu ms", duration);

        return true;
    }

    return false;
}

void BasicRenderer::SceneLoaded()
{
    Super::SceneLoaded();

    m_Scene->FinishedLoading(GetFrameIndex());

    m_WallclockTime = 0.f;
    m_PreviousViewsValid = false;

    for (auto light : m_Scene->GetSceneGraph()->GetLights())
    {
        if (light->GetLightType() == LightType_Directional)
        {
            m_SunLight = std::static_pointer_cast<donut::engine::DirectionalLight>(light);
            if (m_SunLight->irradiance <= 0.f)
                m_SunLight->irradiance = 1.f;
            break;
        }
    }

    if (!m_SunLight)
    {
        m_SunLight = std::make_shared<donut::engine::DirectionalLight>();
        m_SunLight->angularSize = 0.53f;
        m_SunLight->irradiance = 1.f;

        auto node = std::make_shared<donut::engine::SceneGraphNode>();
        node->SetLeaf(m_SunLight);
        m_SunLight->SetDirection(dm::double3(0.1, -0.9, 0.1));
        m_SunLight->SetName("Sun");
        m_Scene->GetSceneGraph()->Attach(m_Scene->GetSceneGraph()->GetRootNode(), node);
    }

    auto cameras = m_Scene->GetSceneGraph()->GetCameras();
    if (!cameras.empty())
    {
        m_ui.ActiveSceneCamera = cameras[0];
    }
    else
    {
        m_ui.ActiveSceneCamera.reset();

        m_FirstPersonCamera.LookAt(
            dm::float3(0.f, 1.8f, 0.f),
            dm::float3(1.f, 1.8f, 0.f));
        m_CameraVerticalFov = 60.f;
    }

    m_ThirdPersonCamera.SetRotation(dm::radians(135.f), dm::radians(20.f));
    PointThirdPersonCameraAt(m_Scene->GetSceneGraph()->GetRootNode());

    m_ui.UseThirdPersonCamera = donut::string_utils::ends_with(m_CurrentSceneName, ".gltf")
        || donut::string_utils::ends_with(m_CurrentSceneName, ".glb");

    CopyActiveCameraToFirstPerson();
}

void BasicRenderer::PointThirdPersonCameraAt(const std::shared_ptr<donut::engine::SceneGraphNode>& node)
{
    dm::box3 bounds = node->GetGlobalBoundingBox();
    m_ThirdPersonCamera.SetTargetPosition(bounds.center());
    float radius = length(bounds.diagonal()) * 0.5f;
    float distance = radius / sinf(dm::radians(m_CameraVerticalFov * 0.5f));
    m_ThirdPersonCamera.SetDistance(distance);
    m_ThirdPersonCamera.Animate(0.f);
}

bool BasicRenderer::IsStereo()
{
    return m_ui.Stereo;
}

std::shared_ptr<donut::engine::TextureCache> BasicRenderer::GetTextureCache()
{
    return m_TextureCache;
}

std::shared_ptr<donut::engine::Scene> BasicRenderer::GetScene()
{
    return m_Scene;
}

bool BasicRenderer::SetupView()
{
    dm::float2 renderTargetSize = dm::float2(m_RenderTargets->GetSize());

    if (m_TemporalAntiAliasingPass)
        m_TemporalAntiAliasingPass->SetJitter(m_ui.TemporalAntiAliasingJitter);

    dm::float2 pixelOffset = (m_ui.AntiAliasingMode == AntiAliasingMode::TEMPORAL ||
                          m_ui.AntiAliasingMode == AntiAliasingMode::DLSS) && m_TemporalAntiAliasingPass
        ? m_TemporalAntiAliasingPass->GetCurrentPixelOffset()
        : dm::float2(0.f);

    std::shared_ptr<donut::engine::StereoPlanarView> stereoView = std::dynamic_pointer_cast<donut::engine::StereoPlanarView, donut::engine::IView>(m_View);
    std::shared_ptr<donut::engine::PlanarView> planarView = std::dynamic_pointer_cast<donut::engine::PlanarView, donut::engine::IView>(m_View);

    dm::affine3 viewMatrix;
    float verticalFov = dm::radians(m_CameraVerticalFov);
    float zNear = 0.01f;
    if (m_ui.ActiveSceneCamera)
    {
        auto perspectiveCamera = std::dynamic_pointer_cast<donut::engine::PerspectiveCamera>(m_ui.ActiveSceneCamera);
        if (perspectiveCamera)
        {
            zNear = perspectiveCamera->zNear;
            verticalFov = perspectiveCamera->verticalFov;
        }

        viewMatrix = m_ui.ActiveSceneCamera->GetWorldToViewMatrix();
    }
    else
    {
        viewMatrix = GetActiveCamera().GetWorldToViewMatrix();
    }

    bool topologyChanged = false;

    if (IsStereo())
    {
        if (!stereoView)
        {
            m_View = stereoView = std::make_shared<donut::engine::StereoPlanarView>();
            m_ViewPrevious = std::make_shared<donut::engine::StereoPlanarView>();
            topologyChanged = true;
        }

        stereoView->LeftView.SetViewport(nvrhi::Viewport(renderTargetSize.x * 0.5f, renderTargetSize.y));
        stereoView->LeftView.SetPixelOffset(pixelOffset);

        stereoView->RightView.SetViewport(nvrhi::Viewport(renderTargetSize.x * 0.5f, renderTargetSize.x, 0.f, renderTargetSize.y, 0.f, 1.f));
        stereoView->RightView.SetPixelOffset(pixelOffset);

        {
            dm::float4x4 projection = donut::math::perspProjD3DStyleReverse(verticalFov, renderTargetSize.x / renderTargetSize.y * 0.5f, zNear);

            dm::affine3 leftView = viewMatrix;
            stereoView->LeftView.SetMatrices(leftView, projection);

            dm::affine3 rightView = leftView;
            rightView.m_translation -= dm::float3(0.2f, 0, 0);
            stereoView->RightView.SetMatrices(rightView, projection);
        }

        stereoView->LeftView.UpdateCache();
        stereoView->RightView.UpdateCache();

        m_ThirdPersonCamera.SetView(stereoView->LeftView);

        if (topologyChanged)
        {
            *std::static_pointer_cast<donut::engine::StereoPlanarView>(m_ViewPrevious) = *std::static_pointer_cast<donut::engine::StereoPlanarView>(m_View);
        }
    }
    else
    {
        if (!planarView)
        {
            m_View = planarView = std::make_shared<donut::engine::PlanarView>();
            m_ViewPrevious = std::make_shared<donut::engine::PlanarView>();
            topologyChanged = true;
        }

        dm::float4x4 projection = donut::math::perspProjD3DStyleReverse(verticalFov, renderTargetSize.x / renderTargetSize.y, zNear);

        planarView->SetViewport(nvrhi::Viewport(renderTargetSize.x, renderTargetSize.y));
        planarView->SetPixelOffset(pixelOffset);

        planarView->SetMatrices(viewMatrix, projection);
        planarView->UpdateCache();

        m_ThirdPersonCamera.SetView(*planarView);

        if (topologyChanged)
        {
            *std::static_pointer_cast<donut::engine::PlanarView>(m_ViewPrevious) = *std::static_pointer_cast<donut::engine::PlanarView>(m_View);
        }
    }

    return topologyChanged;
}

void BasicRenderer::CreateRenderPasses(bool& exposureResetRequired)
{
    uint32_t motionVectorStencilMask = 0x01;

    donut::render::ForwardShadingPass::CreateParameters ForwardParams;
    ForwardParams.trackLiveness = false;
    m_ForwardPass = std::make_unique<donut::render::ForwardShadingPass>(GetDevice(), m_CommonPasses);
    m_ForwardPass->Init(*m_ShaderFactory, ForwardParams);

    donut::render::GBufferFillPass::CreateParameters GBufferParams;
    GBufferParams.enableMotionVectors = true;
    GBufferParams.stencilWriteMask = motionVectorStencilMask;
    m_GBufferPass = std::make_unique<donut::render::GBufferFillPass>(GetDevice(), m_CommonPasses);
    m_GBufferPass->Init(*m_ShaderFactory, GBufferParams);

    GBufferParams.enableMotionVectors = false;
    m_MaterialIDPass = std::make_unique<donut::render::MaterialIDPass>(GetDevice(), m_CommonPasses);
    m_MaterialIDPass->Init(*m_ShaderFactory, GBufferParams);

    m_PixelReadbackPass = std::make_unique<donut::render::PixelReadbackPass>(GetDevice(), m_ShaderFactory, m_RenderTargets->MaterialIDs, nvrhi::Format::RGBA32_UINT);
    m_MipMapGenPass = std::make_unique <donut::render::MipMapGenPass>(GetDevice(), m_ShaderFactory, m_RenderTargets->ResolvedColor, donut::render::MipMapGenPass::Mode::MODE_COLOR);

    m_DeferredLightingPass = std::make_unique<donut::render::DeferredLightingPass>(GetDevice(), m_CommonPasses);
    m_DeferredLightingPass->Init(m_ShaderFactory);

    m_SkyPass = std::make_unique<donut::render::SkyPass>(GetDevice(), m_ShaderFactory, m_CommonPasses, m_RenderTargets->ForwardFramebuffer, *m_View);

    {
        donut::render::TemporalAntiAliasingPass::CreateParameters taaParams;
        taaParams.sourceDepth = m_RenderTargets->Depth;
        taaParams.motionVectors = m_RenderTargets->MotionVectors;
        taaParams.unresolvedColor = m_RenderTargets->HdrColor;
        taaParams.resolvedColor = m_RenderTargets->ResolvedColor;
        taaParams.feedback1 = m_RenderTargets->TemporalFeedback1;
        taaParams.feedback2 = m_RenderTargets->TemporalFeedback2;
        taaParams.motionVectorStencilMask = motionVectorStencilMask;
        taaParams.useCatmullRomFilter = true;

        m_TemporalAntiAliasingPass = std::make_unique<donut::render::TemporalAntiAliasingPass>(GetDevice(), m_ShaderFactory, m_CommonPasses, *m_View, taaParams);
    }

    if (m_RenderTargets->GetSampleCount() == 1)
    {
        m_SsaoPass = std::make_unique<donut::render::SsaoPass>(GetDevice(), m_ShaderFactory, m_CommonPasses, m_RenderTargets->Depth, m_RenderTargets->GBufferNormals, m_RenderTargets->AmbientOcclusion);
    }

    m_LightProbePass = std::make_shared<donut::render::LightProbeProcessingPass>(GetDevice(), m_ShaderFactory, m_CommonPasses);

    nvrhi::BufferHandle exposureBuffer = nullptr;
    if (m_ToneMappingPass)
        exposureBuffer = m_ToneMappingPass->GetExposureBuffer();
    else
        exposureResetRequired = true;

    donut::render::ToneMappingPass::CreateParameters toneMappingParams;
    toneMappingParams.exposureBufferOverride = exposureBuffer;
    m_ToneMappingPass = std::make_unique<donut::render::ToneMappingPass>(GetDevice(), m_ShaderFactory, m_CommonPasses, m_RenderTargets->LdrFramebuffer, *m_View, toneMappingParams);

    m_BloomPass = std::make_unique<donut::render::BloomPass>(GetDevice(), m_ShaderFactory, m_CommonPasses, m_RenderTargets->ResolvedFramebuffer, *m_View);

#if DONUT_WITH_DLSS
    if (m_DLSS)
    {
        dm::uint2 size = m_RenderTargets->GetSize();
        donut::render::DLSS::InitParameters initParameters = {};
        initParameters.inputWidth = size.x;
        initParameters.inputHeight = size.y;
        initParameters.outputWidth = size.x;
        initParameters.outputHeight = size.y;
        m_DLSS->Init(initParameters);

        m_ui.DlssAvailable = m_DLSS->IsDlssInitialized();
    }
#endif

    m_PreviousViewsValid = false;
}

void BasicRenderer::RenderSplashScreen(nvrhi::IFramebuffer* framebuffer)
{
    nvrhi::ITexture* framebufferTexture = framebuffer->getDesc().colorAttachments[0].texture;
    m_CommandList->open();
    m_CommandList->clearTextureFloat(framebufferTexture, nvrhi::AllSubresources, nvrhi::Color(0.f));
    m_CommandList->close();
    GetDevice()->executeCommandList(m_CommandList);
    GetDeviceManager()->SetVsyncEnabled(true);
}

void BasicRenderer::RenderScene(nvrhi::IFramebuffer* framebuffer)
{
    int windowWidth, windowHeight;
    GetDeviceManager()->GetWindowDimensions(windowWidth, windowHeight);
    nvrhi::Viewport windowViewport = nvrhi::Viewport(float(windowWidth), float(windowHeight));
    nvrhi::Viewport renderViewport = windowViewport;

    m_Scene->RefreshSceneGraph(GetFrameIndex());

    bool exposureResetRequired = false;

    {
        dm::uint width = windowWidth;
        dm::uint height = windowHeight;

        dm::uint sampleCount = 1;
        switch (m_ui.AntiAliasingMode)
        {
        case AntiAliasingMode::MSAA_2X: sampleCount = 2; break;
        case AntiAliasingMode::MSAA_4X: sampleCount = 4; break;
        case AntiAliasingMode::MSAA_8X: sampleCount = 8; break;
        default:;
        }

        bool needNewPasses = false;

        if (!m_RenderTargets || m_RenderTargets->IsUpdateRequired(dm::uint2(width, height), sampleCount))
        {
            m_RenderTargets = nullptr;
            m_BindingCache.Clear();
            m_RenderTargets = std::make_unique<RenderTargets>();
            m_RenderTargets->Init(GetDevice(), dm::uint2(width, height), sampleCount, true, true);

            needNewPasses = true;
        }

        if (SetupView())
        {
            needNewPasses = true;
        }

        if (m_ui.ShaderReoladRequested)
        {
            m_ShaderFactory->ClearCache();
            needNewPasses = true;
        }

        if(needNewPasses)
        {
            CreateRenderPasses(exposureResetRequired);
        }

        m_ui.ShaderReoladRequested = false;
    }

    m_CommandList->open();

    m_Scene->RefreshBuffers(m_CommandList, GetFrameIndex());

    nvrhi::ITexture* framebufferTexture = framebuffer->getDesc().colorAttachments[0].texture;
    m_CommandList->clearTextureFloat(framebufferTexture, nvrhi::AllSubresources, nvrhi::Color(0.f));

    m_AmbientTop = m_ui.AmbientIntensity * m_ui.SkyParams.skyColor * m_ui.SkyParams.brightness;
    m_AmbientBottom = m_ui.AmbientIntensity * m_ui.SkyParams.groundColor * m_ui.SkyParams.brightness;
    if (m_ui.EnableShadows)
    {
        m_SunLight->shadowMap = m_ShadowMap;
        dm::box3 sceneBounds = m_Scene->GetSceneGraph()->GetRootNode()->GetGlobalBoundingBox();

        dm::frustum projectionFrustum = m_View->GetProjectionFrustum();
        const float maxShadowDistance = 100.f;

        dm::affine3 viewMatrixInv = m_View->GetChildView(donut::engine::ViewType::PLANAR, 0)->GetInverseViewMatrix();

        float zRange = length(sceneBounds.diagonal());
        m_ShadowMap->SetupForPlanarViewStable(*m_SunLight, projectionFrustum, viewMatrixInv, maxShadowDistance, zRange, zRange, m_ui.CsmExponent);

        m_ShadowMap->Clear(m_CommandList);

        donut::render::DepthPass::Context context;

        RenderCompositeView(m_CommandList,
            &m_ShadowMap->GetView(), nullptr,
            *m_ShadowFramebuffer,
            m_Scene->GetSceneGraph()->GetRootNode(),
            *m_OpaqueDrawStrategy,
            *m_ShadowDepthPass,
            context,
            "ShadowMap",
            m_ui.EnableMaterialEvents);
    }
    else
    {
        m_SunLight->shadowMap = nullptr;
    }

    std::vector<std::shared_ptr<donut::engine::LightProbe>> lightProbes;
    if (m_ui.EnableLightProbe)
    {
        for (auto probe : m_LightProbes)
        {
            if (probe->enabled)
            {
                probe->diffuseScale = m_ui.LightProbeDiffuseScale;
                probe->specularScale = m_ui.LightProbeSpecularScale;
                lightProbes.push_back(probe);
            }
        }
    }

    m_RenderTargets->Clear(m_CommandList);

    if (exposureResetRequired)
        m_ToneMappingPass->ResetExposure(m_CommandList, 0.5f);

    donut::render::ForwardShadingPass::Context forwardContext;

    if (!m_ui.UseDeferredShading || m_ui.EnableTranslucency)
    {
        m_ForwardPass->PrepareLights(forwardContext, m_CommandList, m_Scene->GetSceneGraph()->GetLights(), m_AmbientTop, m_AmbientBottom, lightProbes);
    }

    if (m_ui.UseDeferredShading)
    {
        donut::render::GBufferFillPass::Context gbufferContext;

        RenderCompositeView(m_CommandList,
            m_View.get(), m_ViewPrevious.get(),
            *m_RenderTargets->GBufferFramebuffer,
            m_Scene->GetSceneGraph()->GetRootNode(),
            *m_OpaqueDrawStrategy,
            *m_GBufferPass,
            gbufferContext,
            "GBufferFill",
            m_ui.EnableMaterialEvents);

        nvrhi::ITexture* ambientOcclusionTarget = nullptr;
        if (m_ui.EnableSsao && m_SsaoPass)
        {
            m_SsaoPass->Render(m_CommandList, m_ui.SsaoParams, *m_View);
            ambientOcclusionTarget = m_RenderTargets->AmbientOcclusion;
        }

        donut::render::DeferredLightingPass::Inputs deferredInputs;
        deferredInputs.SetGBuffer(*m_RenderTargets);
        deferredInputs.ambientOcclusion = m_ui.EnableSsao ? m_RenderTargets->AmbientOcclusion : nullptr;
        deferredInputs.ambientColorTop = m_AmbientTop;
        deferredInputs.ambientColorBottom = m_AmbientBottom;
        deferredInputs.lights = &m_Scene->GetSceneGraph()->GetLights();
        deferredInputs.lightProbes = m_ui.EnableLightProbe ? &m_LightProbes : nullptr;
        deferredInputs.output = m_RenderTargets->HdrColor;

        m_DeferredLightingPass->Render(m_CommandList, *m_View, deferredInputs);
    }
    else
    {
        RenderCompositeView(m_CommandList,
            m_View.get(), m_ViewPrevious.get(),
            *m_RenderTargets->ForwardFramebuffer,
            m_Scene->GetSceneGraph()->GetRootNode(),
            *m_OpaqueDrawStrategy,
            *m_ForwardPass,
            forwardContext,
            "ForwardOpaque",
            m_ui.EnableMaterialEvents);
    }

    if(m_Pick)
    {
        m_CommandList->clearTextureUInt(m_RenderTargets->MaterialIDs, nvrhi::AllSubresources, 0xffff);

        donut::render::MaterialIDPass::Context materialIdContext;

        RenderCompositeView(m_CommandList,
            m_View.get(), m_ViewPrevious.get(),
            *m_RenderTargets->MaterialIDFramebuffer,
            m_Scene->GetSceneGraph()->GetRootNode(),
            *m_OpaqueDrawStrategy,
            *m_MaterialIDPass,
            materialIdContext,
            "MaterialID");

        if (m_ui.EnableTranslucency)
        {
            RenderCompositeView(m_CommandList,
                m_View.get(), m_ViewPrevious.get(),
                *m_RenderTargets->MaterialIDFramebuffer,
                m_Scene->GetSceneGraph()->GetRootNode(),
                *m_TransparentDrawStrategy,
                *m_MaterialIDPass,
                materialIdContext,
                "MaterialID - Translucent");
        }

        m_PixelReadbackPass->Capture(m_CommandList, m_PickPosition);
    }

    if (m_ui.EnableProceduralSky)
        m_SkyPass->Render(m_CommandList, *m_View, *m_SunLight, m_ui.SkyParams);

    if (m_ui.EnableTranslucency)
    {
        RenderCompositeView(m_CommandList,
            m_View.get(), m_ViewPrevious.get(),
            *m_RenderTargets->ForwardFramebuffer,
            m_Scene->GetSceneGraph()->GetRootNode(),
            *m_TransparentDrawStrategy,
            *m_ForwardPass,
            forwardContext,
            "ForwardTransparent",
            m_ui.EnableMaterialEvents);
    }

    nvrhi::ITexture* finalHdrColor = m_RenderTargets->HdrColor;

    if (m_ui.AntiAliasingMode == AntiAliasingMode::TEMPORAL || m_ui.AntiAliasingMode == AntiAliasingMode::DLSS)
    {
        if (m_PreviousViewsValid)
        {
            m_TemporalAntiAliasingPass->RenderMotionVectors(m_CommandList, *m_View, *m_ViewPrevious);
        }

#if DONUT_WITH_DLSS
        if (m_ui.AntiAliasingMode == AntiAliasingMode::DLSS)
        {
            if (m_DLSS && m_DLSS->IsDlssInitialized() && !IsStereo())
            {
                auto planarView = std::dynamic_pointer_cast<donut::engine::PlanarView>(m_View);
                assert(planarView);

                donut::render::DLSS::EvaluateParameters params;
                params.depthTexture = m_RenderTargets->Depth;
                params.motionVectorsTexture = m_RenderTargets->MotionVectors;
                params.inputColorTexture = m_RenderTargets->HdrColor;
                params.outputColorTexture = m_RenderTargets->ResolvedColor;
                params.exposureBuffer = m_ToneMappingPass->GetExposureBuffer();

                m_DLSS->Evaluate(m_CommandList, params, *planarView);
            }
            else
            {
                // Fallback to TAA if DLSS is not available
                m_ui.AntiAliasingMode = AntiAliasingMode::TEMPORAL;
            }
        }
#endif

        if (m_ui.AntiAliasingMode == AntiAliasingMode::TEMPORAL)
        {
            m_TemporalAntiAliasingPass->TemporalResolve(m_CommandList, m_ui.TemporalAntiAliasingParams, m_PreviousViewsValid, *m_View, *m_View);
        }

        finalHdrColor = m_RenderTargets->ResolvedColor;

        if (m_ui.EnableBloom)
        {
            m_BloomPass->Render(m_CommandList, m_RenderTargets->ResolvedFramebuffer, *m_View, m_RenderTargets->ResolvedColor, m_ui.BloomSigma, m_ui.BloomAlpha);
        }
        m_PreviousViewsValid = true;
    }
    else
    {
        std::shared_ptr<donut::engine::FramebufferFactory> finalHdrFramebuffer = m_RenderTargets->HdrFramebuffer;

        if (m_RenderTargets->GetSampleCount() > 1)
        {
            auto subresources = nvrhi::TextureSubresourceSet(0, 1, 0, 1);
            m_CommandList->resolveTexture(m_RenderTargets->ResolvedColor, subresources, m_RenderTargets->HdrColor, subresources);
            finalHdrColor = m_RenderTargets->ResolvedColor;
            finalHdrFramebuffer = m_RenderTargets->ResolvedFramebuffer;
        }

        if (m_ui.EnableBloom)
        {
            m_BloomPass->Render(m_CommandList, finalHdrFramebuffer, *m_View, finalHdrColor, m_ui.BloomSigma, m_ui.BloomAlpha);
        }

        m_PreviousViewsValid = false;
    }

    auto toneMappingParams = m_ui.ToneMappingParams;
    if (exposureResetRequired)
    {
        toneMappingParams.eyeAdaptationSpeedUp = 0.f;
        toneMappingParams.eyeAdaptationSpeedDown = 0.f;
    }
    m_ToneMappingPass->SimpleRender(m_CommandList, toneMappingParams, *m_View, finalHdrColor);

    m_CommonPasses->BlitTexture(m_CommandList, framebuffer, m_RenderTargets->LdrColor, &m_BindingCache);

    if (m_ui.TestMipMapGen)
    {
        m_MipMapGenPass->Dispatch(m_CommandList);
        m_MipMapGenPass->Display(m_CommonPasses, m_CommandList, framebuffer);
    }

    if (m_ui.DisplayShadowMap)
    {
        for (int cascade = 0; cascade < 4; cascade++)
        {
            nvrhi::Viewport viewport = nvrhi::Viewport(
                10.f + 266.f * cascade,
                266.f * (1 + cascade),
                windowViewport.maxY - 266.f,
                windowViewport.maxY - 10.f, 0.f, 1.f
            );

            donut::engine::BlitParameters blitParams;
            blitParams.targetFramebuffer = framebuffer;
            blitParams.targetViewport = viewport;
            blitParams.sourceTexture = m_ShadowMap->GetTexture();
            blitParams.sourceArraySlice = cascade;
            m_CommonPasses->BlitTexture(m_CommandList, blitParams, &m_BindingCache);
        }
    }

    m_CommandList->close();
    GetDevice()->executeCommandList(m_CommandList);

    if (!m_ui.ScreenshotFileName.empty())
    {
        SaveTextureToFile(GetDevice(), m_CommonPasses.get(), framebufferTexture, nvrhi::ResourceStates::RenderTarget, m_ui.ScreenshotFileName.c_str());
        m_ui.ScreenshotFileName = "";
    }

    if (m_Pick)
    {
        m_Pick = false;
        dm::uint4 pixelValue = m_PixelReadbackPass->ReadUInts();
        m_ui.SelectedMaterial = nullptr;
        m_ui.SelectedNode = nullptr;

        for (const auto& material : m_Scene->GetSceneGraph()->GetMaterials())
        {
            if (material->materialID == int(pixelValue.x))
            {
                m_ui.SelectedMaterial = material;
                break;
            }
        }

        for (const auto& instance : m_Scene->GetSceneGraph()->GetMeshInstances())
        {
            if (instance->GetInstanceIndex() == int(pixelValue.y))
            {
                m_ui.SelectedNode = instance->GetNodeSharedPtr();
                break;
            }
        }

        if (m_ui.SelectedNode)
        {
            Log(Info, "Picked node: %s", m_ui.SelectedNode->GetPath().generic_string().c_str());
            PointThirdPersonCameraAt(m_ui.SelectedNode);
        }
        else
        {
            PointThirdPersonCameraAt(m_Scene->GetSceneGraph()->GetRootNode());
        }
    }

    m_TemporalAntiAliasingPass->AdvanceFrame();
    std::swap(m_View, m_ViewPrevious);

    GetDeviceManager()->SetVsyncEnabled(m_ui.EnableVsync);
}

std::shared_ptr<donut::engine::ShaderFactory> BasicRenderer::GetShaderFactory()
{
    return m_ShaderFactory;
}

std::vector<std::shared_ptr<donut::engine::LightProbe>>& BasicRenderer::GetLightProbes()
{
    return m_LightProbes;
}

void BasicRenderer::CreateLightProbes(uint32_t numProbes)
{
    nvrhi::DeviceHandle device = GetDeviceManager()->GetDevice();

    uint32_t diffuseMapSize = 256;
    uint32_t diffuseMapMipLevels = 1;
    uint32_t specularMapSize = 512;
    uint32_t specularMapMipLevels = 8;

    nvrhi::TextureDesc cubemapDesc;

    cubemapDesc.arraySize = 6 * numProbes;
    cubemapDesc.dimension = nvrhi::TextureDimension::TextureCubeArray;
    cubemapDesc.isRenderTarget = true;
    cubemapDesc.keepInitialState = true;

    cubemapDesc.width = diffuseMapSize;
    cubemapDesc.height = diffuseMapSize;
    cubemapDesc.mipLevels = diffuseMapMipLevels;
    cubemapDesc.format = nvrhi::Format::RGBA16_FLOAT;
    cubemapDesc.initialState = nvrhi::ResourceStates::ShaderResource;
    cubemapDesc.keepInitialState = true;

    m_LightProbeDiffuseTexture = device->createTexture(cubemapDesc);

    cubemapDesc.width = specularMapSize;
    cubemapDesc.height = specularMapSize;
    cubemapDesc.mipLevels = specularMapMipLevels;
    cubemapDesc.format = nvrhi::Format::RGBA16_FLOAT;
    cubemapDesc.initialState = nvrhi::ResourceStates::ShaderResource;
    cubemapDesc.keepInitialState = true;

    m_LightProbeSpecularTexture = device->createTexture(cubemapDesc);

    m_LightProbes.clear();

    for (uint32_t i = 0; i < numProbes; i++)
    {
        std::shared_ptr<donut::engine::LightProbe> probe = std::make_shared<donut::engine::LightProbe>();

        probe->name = std::to_string(i + 1);
        probe->diffuseMap = m_LightProbeDiffuseTexture;
        probe->specularMap = m_LightProbeSpecularTexture;
        probe->diffuseArrayIndex = i;
        probe->specularArrayIndex = i;
        probe->bounds = dm::frustum::empty();
        probe->enabled = false;

        m_LightProbes.push_back(probe);
    }
}

void BasicRenderer::RenderLightProbe(donut::engine::LightProbe& probe)
{
    nvrhi::DeviceHandle device = GetDeviceManager()->GetDevice();

    uint32_t environmentMapSize = 1024;
    uint32_t environmentMapMipLevels = 8;

    nvrhi::TextureDesc cubemapDesc;
    cubemapDesc.arraySize = 6;
    cubemapDesc.width = environmentMapSize;
    cubemapDesc.height = environmentMapSize;
    cubemapDesc.mipLevels = environmentMapMipLevels;
    cubemapDesc.dimension = nvrhi::TextureDimension::TextureCube;
    cubemapDesc.isRenderTarget = true;
    cubemapDesc.format = nvrhi::Format::RGBA16_FLOAT;
    cubemapDesc.initialState = nvrhi::ResourceStates::RenderTarget;
    cubemapDesc.keepInitialState = true;
    cubemapDesc.clearValue = nvrhi::Color(0.f);
    cubemapDesc.useClearValue = true;

    nvrhi::TextureHandle colorTexture = device->createTexture(cubemapDesc);

    const nvrhi::Format depthFormats[] = {
        nvrhi::Format::D24S8,
        nvrhi::Format::D32,
        nvrhi::Format::D16,
        nvrhi::Format::D32S8 };

    const nvrhi::FormatSupport depthFeatures =
        nvrhi::FormatSupport::Texture |
        nvrhi::FormatSupport::DepthStencil |
        nvrhi::FormatSupport::ShaderLoad;

    cubemapDesc.mipLevels = 1;
    cubemapDesc.format = nvrhi::utils::ChooseFormat(GetDevice(), depthFeatures, depthFormats, std::size(depthFormats));
    cubemapDesc.isTypeless = true;
    cubemapDesc.initialState = nvrhi::ResourceStates::DepthWrite;

    nvrhi::TextureHandle depthTexture = device->createTexture(cubemapDesc);

    std::shared_ptr<donut::engine::FramebufferFactory> framebuffer = std::make_shared<donut::engine::FramebufferFactory>(device);
    framebuffer->RenderTargets = { colorTexture };
    framebuffer->DepthTarget = depthTexture;

    donut::engine::CubemapView view;
    view.SetArrayViewports(environmentMapSize, 0);
    const float nearPlane = 0.1f;
    const float cullDistance = 100.f;
    dm::float3 probePosition = GetActiveCamera().GetPosition();
    if (m_ui.ActiveSceneCamera)
        probePosition = m_ui.ActiveSceneCamera->GetWorldToViewMatrix().m_translation;

    view.SetTransform(dm::translation(-probePosition), nearPlane, cullDistance);
    view.UpdateCache();

    std::shared_ptr<donut::render::SkyPass> skyPass = std::make_shared<donut::render::SkyPass>(device, m_ShaderFactory, m_CommonPasses, framebuffer, view);

    donut::render::ForwardShadingPass::CreateParameters ForwardParams;
    ForwardParams.singlePassCubemap = GetDevice()->queryFeatureSupport(nvrhi::Feature::FastGeometryShader);
    std::shared_ptr<donut::render::ForwardShadingPass> forwardPass = std::make_shared<donut::render::ForwardShadingPass>(device, m_CommonPasses);
    forwardPass->Init(*m_ShaderFactory, ForwardParams);

    nvrhi::CommandListHandle commandList = device->createCommandList();
    commandList->open();
    commandList->clearTextureFloat(colorTexture, nvrhi::AllSubresources, nvrhi::Color(0.f));

    const nvrhi::FormatInfo& depthFormatInfo = nvrhi::getFormatInfo(depthTexture->getDesc().format);
    commandList->clearDepthStencilTexture(depthTexture, nvrhi::AllSubresources, true, 0.f, depthFormatInfo.hasStencil, 0);

    dm::box3 sceneBounds = m_Scene->GetSceneGraph()->GetRootNode()->GetGlobalBoundingBox();
    float zRange = length(sceneBounds.diagonal()) * 0.5f;
    m_ShadowMap->SetupForCubemapView(*m_SunLight, view.GetViewOrigin(), cullDistance, zRange, zRange, m_ui.CsmExponent);
    m_ShadowMap->Clear(commandList);

    donut::render::DepthPass::Context shadowContext;

    RenderCompositeView(commandList,
        &m_ShadowMap->GetView(), nullptr,
        *m_ShadowFramebuffer,
        m_Scene->GetSceneGraph()->GetRootNode(),
        *m_OpaqueDrawStrategy,
        *m_ShadowDepthPass,
        shadowContext,
        "ShadowMap");

    donut::render::ForwardShadingPass::Context forwardContext;

    std::vector<std::shared_ptr<donut::engine::LightProbe>> lightProbes;
    forwardPass->PrepareLights(forwardContext, commandList, m_Scene->GetSceneGraph()->GetLights(), m_AmbientTop, m_AmbientBottom, lightProbes);

    RenderCompositeView(commandList,
        &view, nullptr,
        *framebuffer,
        m_Scene->GetSceneGraph()->GetRootNode(),
        *m_OpaqueDrawStrategy,
        *forwardPass,
        forwardContext,
        "ForwardOpaque");

    skyPass->Render(commandList, view, *m_SunLight, m_ui.SkyParams);

    RenderCompositeView(commandList,
        &view, nullptr,
        *framebuffer,
        m_Scene->GetSceneGraph()->GetRootNode(),
        *m_TransparentDrawStrategy,
        *forwardPass,
        forwardContext,
        "ForwardTransparent");

    m_LightProbePass->GenerateCubemapMips(commandList, colorTexture, 0, 0, environmentMapMipLevels - 1);

    m_LightProbePass->RenderDiffuseMap(commandList, colorTexture, nvrhi::AllSubresources, probe.diffuseMap, probe.diffuseArrayIndex * 6, 0);

    uint32_t specularMapMipLevels = probe.specularMap->getDesc().mipLevels;
    for (uint32_t mipLevel = 0; mipLevel < specularMapMipLevels; mipLevel++)
    {
        float roughness = powf(float(mipLevel) / float(specularMapMipLevels - 1), 2.0f);
        m_LightProbePass->RenderSpecularMap(commandList, roughness, colorTexture, nvrhi::AllSubresources, probe.specularMap, probe.specularArrayIndex * 6, mipLevel);
    }

    m_LightProbePass->RenderEnvironmentBrdfTexture(commandList);

    commandList->close();
    device->executeCommandList(commandList);
    device->waitForIdle();
    device->runGarbageCollection();

    probe.environmentBrdf = m_LightProbePass->GetEnvironmentBrdfTexture();
    dm::box3 bounds = dm::box3(probePosition, probePosition).grow(10.f);
    probe.bounds = dm::frustum::fromBox(bounds);
    probe.enabled = true;
}

void BasicRenderer::BuildUI()
{
    if (!m_ui.ShowUI)
        return;

    const auto& io = ImGui::GetIO();

    int width, height;
    GetDeviceManager()->GetWindowDimensions(width, height);

    if (IsSceneLoading())
    {
        return;
    }

    if (m_ui.ShowConsole && m_console)
    {
        m_console->Render(&m_ui.ShowConsole);
    }

    float const fontSize = ImGui::GetFontSize();

    //ImGui::SetNextWindowPos(ImVec2(fontSize * 0.6f, fontSize * 0.6f), 0);
    //ImGui::Begin("Settings", 0, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("Renderer: %s", GetDeviceManager()->GetRendererString());
    double frameTime = GetDeviceManager()->GetAverageFrameTimeSeconds();
    if (frameTime > 0.0)
        ImGui::Text("%.3f ms/frame (%.1f FPS)", frameTime * 1e3, 1.0 / frameTime);

    const std::string sceneDir = GetSceneDir().generic_string();

    auto getRelativePath = [&sceneDir](std::string const& name)
    {
        return donut::string_utils::starts_with(name, sceneDir)
            ? name.c_str() + sceneDir.size()
            : name.c_str();
    };

    const std::string currentScene = GetCurrentSceneName();
    if (ImGui::BeginCombo("Scene", getRelativePath(currentScene)))
    {
        const std::vector<std::string>& scenes = GetAvailableScenes();
        for (const std::string& scene : scenes)
        {
            bool is_selected = scene == currentScene;
            if (ImGui::Selectable(getRelativePath(scene), is_selected))
                SetCurrentSceneName(scene);
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    if (ImGui::Button("Reload Shaders"))
        m_ui.ShaderReoladRequested = true;

    ImGui::Checkbox("VSync", &m_ui.EnableVsync);
    ImGui::Checkbox("Deferred Shading", &m_ui.UseDeferredShading);
    if (m_ui.AntiAliasingMode >= BasicRenderer::AntiAliasingMode::MSAA_2X)
        m_ui.UseDeferredShading = false; // Deferred shading doesn't work with MSAA
    ImGui::Checkbox("Stereo", &m_ui.Stereo);
    ImGui::Checkbox("Animations", &m_ui.EnableAnimations);

    if (ImGui::BeginCombo("Camera (T)", m_ui.ActiveSceneCamera ? m_ui.ActiveSceneCamera->GetName().c_str()
            : m_ui.UseThirdPersonCamera ? "Third-Person" : "First-Person"))
    {
        if (ImGui::Selectable("First-Person", !m_ui.ActiveSceneCamera && !m_ui.UseThirdPersonCamera))
        {
            m_ui.ActiveSceneCamera.reset();
            m_ui.UseThirdPersonCamera = false;
        }
        if (ImGui::Selectable("Third-Person", !m_ui.ActiveSceneCamera && m_ui.UseThirdPersonCamera))
        {
            m_ui.ActiveSceneCamera.reset();
            m_ui.UseThirdPersonCamera = true;
            CopyActiveCameraToFirstPerson();
        }
        for (const auto& camera : GetScene()->GetSceneGraph()->GetCameras())
        {
            if (ImGui::Selectable(camera->GetName().c_str(), m_ui.ActiveSceneCamera == camera))
            {
                m_ui.ActiveSceneCamera = camera;
                CopyActiveCameraToFirstPerson();
            }
        }
        ImGui::EndCombo();
    }

    if (m_ui.AntiAliasingMode == BasicRenderer::AntiAliasingMode::DLSS && !m_ui.DlssAvailable)
        m_ui.AntiAliasingMode = BasicRenderer::AntiAliasingMode::TEMPORAL; // Fallback to TAA if DLSS is not available

    const char* aaModes[] = {
        "None",
        "TemporalAA",
        "DLSS",
        "MSAA 2x",
        "MSAA 4x",
        "MSAA 8x"
    };
    int aaMode = static_cast<int>(m_ui.AntiAliasingMode);
    if (ImGui::BeginCombo("AA Mode", aaModes[aaMode]))
    {
        for (int n = 0; n < IM_ARRAYSIZE(aaModes); n++)
        {
            if (n == int(BasicRenderer::AntiAliasingMode::DLSS) && !m_ui.DlssAvailable)
                continue; // Skip DLSS if not available

            bool isSelected = (aaMode == n);
            if (ImGui::Selectable(aaModes[n], isSelected))
                m_ui.AntiAliasingMode = static_cast<BasicRenderer::AntiAliasingMode>(n);
            if (isSelected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    ImGui::Combo("TAA Camera Jitter", (int*)&m_ui.TemporalAntiAliasingJitter, "MSAA\0Halton\0R2\0White Noise\0");

    ImGui::SliderFloat("Ambient Intensity", &m_ui.AmbientIntensity, 0.f, 1.f);

    ImGui::Checkbox("Enable Light Probe", &m_ui.EnableLightProbe);
    if (m_ui.EnableLightProbe && ImGui::CollapsingHeader("Light Probe"))
    {
        ImGui::DragFloat("Diffuse Scale", &m_ui.LightProbeDiffuseScale, 0.01f, 0.0f, 10.0f);
        ImGui::DragFloat("Specular Scale", &m_ui.LightProbeSpecularScale, 0.01f, 0.0f, 10.0f);
    }

    ImGui::Checkbox("Enable Procedural Sky", &m_ui.EnableProceduralSky);
    if (m_ui.EnableProceduralSky && ImGui::CollapsingHeader("Sky Parameters"))
    {
        ImGui::SliderFloat("Brightness", &m_ui.SkyParams.brightness, 0.f, 1.f);
        ImGui::SliderFloat("Glow Size", &m_ui.SkyParams.glowSize, 0.f, 90.f);
        ImGui::SliderFloat("Glow Sharpness", &m_ui.SkyParams.glowSharpness, 1.f, 10.f);
        ImGui::SliderFloat("Glow Intensity", &m_ui.SkyParams.glowIntensity, 0.f, 1.f);
        ImGui::SliderFloat("Horizon Size", &m_ui.SkyParams.horizonSize, 0.f, 90.f);
    }
    ImGui::Checkbox("Enable SSAO", &m_ui.EnableSsao);
    ImGui::Checkbox("Enable Bloom", &m_ui.EnableBloom);
    ImGui::DragFloat("Bloom Sigma", &m_ui.BloomSigma, 0.01f, 0.1f, 100.f);
    ImGui::DragFloat("Bloom Alpha", &m_ui.BloomAlpha, 0.01f, 0.01f, 1.0f);
    ImGui::Checkbox("Enable Shadows", &m_ui.EnableShadows);
    ImGui::Checkbox("Enable Translucency", &m_ui.EnableTranslucency);

    ImGui::Separator();
    ImGui::Checkbox("Temporal AA Clamping", &m_ui.TemporalAntiAliasingParams.enableHistoryClamping);
    ImGui::Checkbox("Material Events", &m_ui.EnableMaterialEvents);
    ImGui::Separator();

    const auto& lights = GetScene()->GetSceneGraph()->GetLights();

    if (!lights.empty() && ImGui::CollapsingHeader("Lights"))
    {
        if (ImGui::BeginCombo("Select Light", m_SelectedLight ? m_SelectedLight->GetName().c_str() : "(None)"))
        {
            for (const auto& light : lights)
            {
                bool selected = m_SelectedLight == light;
                ImGui::Selectable(light->GetName().c_str(), &selected);
                if (selected)
                {
                    m_SelectedLight = light;
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        if (m_SelectedLight)
        {
            donut::app::LightEditor(*m_SelectedLight);
        }
    }

    ImGui::TextUnformatted("Render Light Probe: ");
    uint32_t probeIndex = 1;
    for (auto probe : GetLightProbes())
    {
        ImGui::SameLine();
        if (ImGui::Button(probe->name.c_str()))
        {
            RenderLightProbe(*probe);
        }
    }

    if (ImGui::Button("Screenshot"))
    {
        std::string fileName;
        if (donut::app::FileDialog(false, "BMP files\0*.bmp\0All files\0*.*\0\0", fileName))
        {
            m_ui.ScreenshotFileName = fileName;
        }
    }

    ImGui::Separator();
    ImGui::Checkbox("Test MipMapGen Pass", &m_ui.TestMipMapGen);
    ImGui::Checkbox("Display Shadow Map", &m_ui.DisplayShadowMap);

    auto material = m_ui.SelectedMaterial;
    if (material)
    {
        ImGui::SetNextWindowPos(ImVec2(float(width) - fontSize * 0.6f, fontSize * 0.6f), 0, ImVec2(1.f, 0.f));
        ImGui::Begin("Material Editor");
        ImGui::Text("Material %d: %s", material->materialID, material->name.c_str());

        donut::engine::MaterialDomain previousDomain = material->domain;
        material->dirty = donut::app::MaterialEditor(material.get(), true);

        if (previousDomain != material->domain)
            GetScene()->GetSceneGraph()->GetRootNode()->InvalidateContent();

        ImGui::End();
    }

    if (m_ui.AntiAliasingMode != BasicRenderer::AntiAliasingMode::NONE &&
        m_ui.AntiAliasingMode != BasicRenderer::AntiAliasingMode::TEMPORAL &&
        m_ui.AntiAliasingMode != BasicRenderer::AntiAliasingMode::DLSS)
        m_ui.UseDeferredShading = false;

    if (!m_ui.UseDeferredShading)
        m_ui.EnableSsao = false;
}

