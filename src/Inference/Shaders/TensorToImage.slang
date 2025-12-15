#include <donut/shaders/binding_helpers.hlsli>

#include "ImageTransform.h"

DECLARE_CBUFFER(ImageTransformConstants, gConstants, 0, 0);

StructuredBuffer<half> inputTensor : REGISTER_SRV(0, 0);
RWTexture2D<float4> outputImage : REGISTER_UAV(0, 0);

[shader("compute")]
[numthreads(16, 16, 1)]
void TensorToImage_cs(uint3 pixelIndex : SV_DispatchThreadID)
{
	if (any(pixelIndex.xy >= gConstants.tensorShape))
		return;

	// [B, C, H, w] -> [B, H, W, C]
	uint linearIndex = pixelIndex.x + pixelIndex.y * gConstants.tensorShape.x;
	uint numPixels = gConstants.tensorShape.x * gConstants.tensorShape.y;

	// Assuming the tensor output has 3 channels (RGB) in [0, 255] range
	float4 pixel = float4(inputTensor[linearIndex], inputTensor[linearIndex + numPixels], inputTensor[linearIndex + 2 * numPixels], 255.0f);

	outputImage[pixelIndex.xy] = pixel / 255.0f;
}
