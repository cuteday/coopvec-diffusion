#include <string>
#include "Inference.h"
#include "Logger.h"

NAMESPACE_BEGIN(fluxel)

void TRTLogger::log(nvinfer1::ILogger::Severity severity,
                    const char *msg) noexcept {
	auto convertSeverity = [](nvinfer1::ILogger::Severity severity) -> Log::Level {
		switch (severity) {
			case nvinfer1::ILogger::Severity::kVERBOSE: return Log::Level::Debug;
			case nvinfer1::ILogger::Severity::kINFO: return Log::Level::Info;
			case nvinfer1::ILogger::Severity::kWARNING: return Log::Level::Warning;
			case nvinfer1::ILogger::Severity::kERROR: return Log::Level::Error;
			case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return Log::Level::Fatal;
			default: return Log::Level::None;
		}
	};
	
	Logger::log(convertSeverity(severity), "[TRT-RTX] " + std::string(msg));
}

NAMESPACE_END(fluxel)