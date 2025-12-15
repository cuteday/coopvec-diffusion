#pragma once

struct ImageTransformConstants {
	uint2 tensorShape;	/* [width, height] */
};

struct ImageWarpConstants {
	uint2 tensorShape;	/* [width, height] */
	uint2 mvecShape;	/* [width, height], used in case that motion vector texture is in a different size than the input image */
	bool enableWarp;	/* whether to warp when sampling the previous output */
};
