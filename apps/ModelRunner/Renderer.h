#pragma once

#include "plugins/BasicRenderer/Renderer.h"
#include "plugins/NeuralInference/NeuralInferencePass.h"

using namespace fluxel;

class ModelRunnerRenderer : public BasicRenderer {
public:
	ModelRunnerRenderer(donut::app::DeviceManager *deviceManager, const std::string& sceneName);
	~ModelRunnerRenderer();

	void RenderScene(nvrhi::IFramebuffer *framebuffer) override;
	void BuildUI() override;
	void SceneLoaded() override;

protected:
	std::unique_ptr<NeuralInferencePass> m_neuralInferencePass;
};
