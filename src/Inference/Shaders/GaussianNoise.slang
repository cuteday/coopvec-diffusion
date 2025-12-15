#include <donut/shaders/binding_helpers.hlsli>

#include "Math/PCGSampler.slang"
#include "ImageTransform.h"

DECLARE_CBUFFER(ImageTransformConstants, gConstants, 0, 0);

RWStructuredBuffer<half> outputTensor : REGISTER_UAV(0, 0);

float2 sampleNormal(inout PCG32 pcg, float mean, float sigma)
{
    // u1 in (0,1] to avoid log(0)
    float u1 = max(pcg.NextFloat(), 1e-6);
    float u2 = pcg.NextFloat();

    float r = sqrt(-2.0 * log(u1));
    float theta = 6.28318530718 * u2;   // 2π * u2

    float2 z = float2(r * cos(theta), r * sin(theta)); // N(0,1) pair
    return z * sigma + mean;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void GaussianNoise_cs(uint3 pixelIndex : SV_DispatchThreadID)
{
	uint globalIndex = pixelIndex.x;
	uint numElements = gConstants.tensorShape.x * gConstants.tensorShape.y;

	if (globalIndex >= numElements)
		return;

	PCG32 pcg = PCG32(0ULL /* random seed */, globalIndex /* sequence index */);

	float3 noise;
	noise.xy = sampleNormal(pcg, 0.0f, 1.0f);
	noise.z = sampleNormal(pcg, 0.0f, 1.0f).x;
	outputTensor[globalIndex * 3 + 0] = half(noise.x);
	outputTensor[globalIndex * 3 + 1] = half(noise.y);
	outputTensor[globalIndex * 3 + 2] = half(noise.z);
}
