#include <donut/shaders/binding_helpers.hlsli>

#include "ImageTransform.h"

DECLARE_CBUFFER(ImageTransformConstants, gConstants, 0, 0);

Texture2D<float4> inputTexture : REGISTER_SRV(0, 0);
RWStructuredBuffer<TensorType> outputTensor : REGISTER_UAV(0, 0);

SamplerState textureSampler : REGISTER_SAMPLER(0, 0);

[shader("compute")]
[numthreads(16, 16, 1)]
void ImageToTensor_cs(uint3 pixelIndex : SV_DispatchThreadID)
{
	if (any(pixelIndex.xy >= gConstants.tensorShape.xy))
		return;

	// [B, H, W, C] -> [B, C, H, w]
	uint linearIndex = pixelIndex.x + pixelIndex.y * gConstants.tensorShape.x;
	uint numPixels = gConstants.tensorShape.x * gConstants.tensorShape.y;

	float2 texCoords = (float2(pixelIndex.xy) + 0.5) / float2(gConstants.tensorShape);
	float3 pixel = inputTexture.SampleLevel(textureSampler, texCoords, 0).rgb * gConstants.scaleFactor;

	outputTensor[linearIndex] = TensorType(pixel.r);
	outputTensor[linearIndex + numPixels] = TensorType(pixel.g);
	outputTensor[linearIndex + 2 * numPixels] = TensorType(pixel.b);
}
