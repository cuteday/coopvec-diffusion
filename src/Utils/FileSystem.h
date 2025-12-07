#pragma once

#include <donut/app/ApplicationBase.h>
#include <filesystem>

#include "Fluxel.h"

NAMESPACE_BEGIN(fluxel)
NAMESPACE_BEGIN(file)

std::filesystem::path projectDir();
std::filesystem::path buildDir();

NAMESPACE_END(file)
NAMESPACE_END(fluxel)
