#pragma once

struct ImageTransformConstants {
	uint2 tensorShape; /* [width, height] */
	float scaleFactor; /* scale factor for the input/output tensors of the model */
};

struct ImageWarpConstants {
	ImageTransformConstants transform;
	uint2 mvecShape;	/* [width, height], used in case that motion vector texture is in a different size than the input image */
	bool enableWarp;	/* whether to warp when sampling the previous output */
};
