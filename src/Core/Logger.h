#pragma once

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "fluxel.h"

#ifdef _WIN32   // for ansi control sequences
#   include <Windows.h>
#else
#   include <sys/ioctl.h>
#   include <unistd.h>
#endif

NAMESPACE_BEGIN(fluxel)

template <typename... Args> 
inline std::string formatString(const std::string &format, Args&& ...args) {
	int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
	if (size_s <= 0) throw std::runtime_error("Error during formatting.");
	auto size = static_cast<size_t>(size_s);
	auto buf = std::make_unique<char[]>(size);
	std::snprintf(buf.get(), size, format.c_str(), args...);
	return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

namespace Log {

	const std::string TERMINAL_ESC = "\033";
	const std::string TERMINAL_RED = "\033[0;31m";
	const std::string TERMINAL_GREEN = "\033[0;32m";
	const std::string TERMINAL_LIGHT_GREEN = "\033[1;32m";
	const std::string TERMINAL_YELLOW = "\033[1;33m";
	const std::string TERMINAL_BLUE = "\033[0;34m";
	const std::string TERMINAL_LIGHT_BLUE = "\033[1;34m";
	const std::string TERMINAL_RESET = "\033[0m";
	const std::string TERMINAL_DEFAULT = TERMINAL_RESET;
	const std::string TERMINAL_BOLD = "\033[1;1m";
	const std::string TERMINAL_LIGHT_RED = "\033[1;31m";

	enum class Level {
		None,
		Fatal,
		Error,
		Warning,
		Success,
		Info,
		Debug,
	};
}

class Logger {
private:
	using Level = Log::Level;
	
	friend void logDebug(const std::string& msg);
	friend void logInfo(const std::string& msg);
	friend void logSuccess(const std::string& msg);
	friend void logWarning(const std::string& msg);
	friend void logError(const std::string& msg, bool terminate);
	friend void logFatal(const std::string& msg, bool terminate);
	template <typename... Args> 
	friend void logMessage(Log::Level level, const std::string &msg, Args &&...payload);
public:
	static void log(Level level, const std::string& msg, bool terminate = false);
};
#ifndef KRR_DEBUG_BUILD
inline void logDebug(const std::string &msg) {}	
#else
inline void logDebug(const std::string& msg) { Logger::log(Logger::Level::Debug, msg); }
#endif
inline void logInfo(const std::string& msg) { Logger::log(Logger::Level::Info, msg); }
inline void logSuccess(const std::string& msg) { Logger::log(Logger::Level::Success, msg); }
inline void logWarning(const std::string& msg) { Logger::log(Logger::Level::Warning, msg); }
inline void logError(const std::string& msg, bool terminate = false) { Logger::log(Logger::Level::Error, msg, terminate); }
inline void logFatal(const std::string& msg, bool terminate = true) { Logger::log(Logger::Level::Fatal, msg, terminate); }

template <typename ...Args> 
inline void logMessage(Log::Level level, const std::string &msg, Args &&...args) {
	Logger::log(level, formatString(msg, std::forward<Args>(args)...));
}

#define Log(level, fmt, ...) do{						\
		log##level(formatString(fmt, ##__VA_ARGS__));	\
	} while (0)

namespace Log {
	inline std::string timeToString(const std::string& fmt, time_t time) {
		char timeStr[128];
		if (std::strftime(timeStr, 128, fmt.c_str(), localtime(&time)) == 0) {
			throw std::runtime_error{ "Could not render local time." };
		}
		return timeStr;
	}

	inline std::string nowToString(const std::string& fmt) {
		return timeToString(fmt, std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
	}
}

NAMESPACE_END(fluxel)
