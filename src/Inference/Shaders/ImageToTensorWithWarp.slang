#include <donut/shaders/binding_helpers.hlsli>

#include "ImageTransform.h"

DECLARE_CBUFFER(ImageWarpConstants, gConstants, 0, 0);

Texture2D<float4> inputTexture : REGISTER_SRV(0, 0);
Texture2D<float2> motionVectorTexture : REGISTER_SRV(1, 0);

RWStructuredBuffer<half> outputTensor : REGISTER_UAV(0, 0);

SamplerState textureSampler : REGISTER_SAMPLER(0, 0);

[shader("compute")]
[numthreads(16, 16, 1)]
void ImageToTensorWithWarp_cs(uint3 pixelIndex : SV_DispatchThreadID)
{
	if (any(pixelIndex.xy >= gConstants.tensorShape.xy))
		return;

	// [B, H, W, C] -> [B, C, H, w]
	uint linearIndex = pixelIndex.x + pixelIndex.y * gConstants.tensorShape.x;
	uint numPixels = gConstants.tensorShape.x * gConstants.tensorShape.y;

	float2 texCoords = (float2(pixelIndex.xy) + 0.5) / float2(gConstants.tensorShape);

	// Warp the texture coordinates if the warp is enabled
	if (gConstants.enableWarp) {
		float2 motionVector = motionVectorTexture.SampleLevel(textureSampler, texCoords, 0).rg;
		// Normalize both coordinates before applying the motion vector (in case that the motion vector texture is in a different size than the input image)
		texCoords += motionVector / float2(gConstants.mvecShape);
		texCoords = clamp(texCoords, float2(0.0f), float2(1.0f));
	}

	float3 pixel = inputTexture.SampleLevel(textureSampler, texCoords, 0).rgb * 255.0f;

	outputTensor[linearIndex] = half(pixel.r);
	outputTensor[linearIndex + numPixels] = half(pixel.g);
	outputTensor[linearIndex + 2 * numPixels] = half(pixel.b);
}
