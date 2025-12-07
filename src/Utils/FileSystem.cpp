#include "FileSystem.h"
#include "Config.h"

NAMESPACE_BEGIN(fluxel)
NAMESPACE_BEGIN(file)

std::filesystem::path projectDir() {
	return FLUXEL_PROJECT_DIR;
}

std::filesystem::path buildDir() {
	return FLUXEL_BUILD_DIR;
}

NAMESPACE_END(file)
NAMESPACE_END(fluxel)
