#include "Renderer.h"

using namespace fluxel;

ModelRunnerRenderer::ModelRunnerRenderer(donut::app::DeviceManager *deviceManager, const std::string& sceneName) : BasicRenderer(deviceManager, sceneName) {
	m_neuralInferencePass = std::make_unique<NeuralInferencePass>(deviceManager);
}

ModelRunnerRenderer::~ModelRunnerRenderer() {
}

void ModelRunnerRenderer::RenderScene(nvrhi::IFramebuffer *framebuffer) {
	BasicRenderer::RenderScene(framebuffer);
	if (m_neuralInferencePass->IsEnabled()) {
		m_neuralInferencePass->Render(framebuffer, m_RenderTargets->MotionVectors);
	}
}

void ModelRunnerRenderer::BuildUI() {
	BasicRenderer::BuildUI();
	ImGui::Text("Model Runner");
	ImGui::Separator();
	bool enableModelRunner = m_neuralInferencePass->IsEnabled();
	if (ImGui::Checkbox("Enable Model Runner", &enableModelRunner)) {
		m_neuralInferencePass->SetEnabled(enableModelRunner);
	}
	if (enableModelRunner) {
		m_neuralInferencePass->BuildUI();
	}
}

void ModelRunnerRenderer::SceneLoaded() {
	BasicRenderer::SceneLoaded();
	CopyActiveCameraToFirstPerson();
	// For now we force the camera to be in first-person mode for interactive demonstration purposes.
	m_ui.UseThirdPersonCamera = false;
	m_ui.ActiveSceneCamera.reset();
}
