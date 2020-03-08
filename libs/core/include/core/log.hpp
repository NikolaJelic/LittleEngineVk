#pragma once
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <fmt/format.h>
#include "core/os.hpp"
#include "core/std_types.hpp"
#include "core/time.hpp"
#if defined(LEVK_DEBUG)
#include <stdexcept>
#include "core/assert.hpp"
#endif

#if defined(LEVK_DEBUG)
/**
 * Variable     : LEVK_DEBUG_LOG
 * Description  : Enables LOG_D and LOGIF_D macros (Level::Debug)
 */
#if !defined(LEVK_LOG_DEBUG)
#define LEVK_LOG_DEBUG
#endif
/**
 * Variable     : LEVK_LOG_SOURCE_LOCATION
 * Description  : Appends source file and line number to log output
 */
#if !defined(LEVK_LOG_SOURCE_LOCATION)
#define LEVK_LOG_SOURCE_LOCATION
#endif
/**
 * Variable     : LEVK_LOG_FORMAT_EXCEPTIONS
 * Description  : Encloses fmt::format(...) in a try-catch block and calls ASSERT on a runtime exception
 */
#if !defined(LEVK_LOG_FORMAT_EXCEPTIONS)
#define LEVK_LOG_FORMAT_EXCEPTIONS
#endif
#endif

#define LOG(level, text, ...) le::log::fmtLog(level, text, __FILE__, __LINE__, ##__VA_ARGS__)
#define LOGIF(predicate, level, text, ...)                               \
	if (predicate)                                                       \
	{                                                                    \
		le::log::fmtLog(level, text, __FILE__, __LINE__, ##__VA_ARGS__); \
	}
#define LOG_E(text, ...) LOG(le::log::Level::Error, text, ##__VA_ARGS__)
#define LOGIF_E(predicate, text, ...) LOGIF(predicate, le::log::Level::Error, text, ##__VA_ARGS__)
#define LOG_W(text, ...) LOG(le::log::Level::Warning, text, ##__VA_ARGS__)
#define LOGIF_W(predicate, text, ...) LOGIF(predicate, le::log::Level::Warning, text, ##__VA_ARGS__)
#define LOG_I(text, ...) LOG(le::log::Level::Info, text, ##__VA_ARGS__)
#define LOGIF_I(predicate, text, ...) LOGIF(predicate, le::log::Level::Info, text, ##__VA_ARGS__)

#if defined(LEVK_LOG_DEBUG)
#define LOG_D(text, ...) LOG(le::log::Level::Debug, text, ##__VA_ARGS__)
#define LOGIF_D(predicate, text, ...) LOGIF(predicate, le::log::Level::Debug, text, ##__VA_ARGS__)
#else
#define LOG_D(text, ...)
#define LOGIF_D(predicate, text, ...)
#endif

namespace le::log
{
enum class Level : u8
{
	Debug = 0,
	Info,
	Warning,
	Error,
	COUNT_
};

struct Service final
{
	Service(std::filesystem::path const& path, Time pollRate = Time::from_s(0.5f));
	~Service();
};

inline Level g_minLevel = Level::Debug;

void logText(Level level, std::string text, std::string_view file, u64 line);

template <typename... Args>
void fmtLog(Level level, std::string_view text, std::string_view file, u64 line, Args... args)
{
	if ((u8)level >= (u8)g_minLevel)
	{
#if defined(LEVK_LOG_FORMAT_EXCEPTIONS)
		try
#endif
		{
			logText(level, fmt::format(text, args...), file, line);
		}
#if defined(LEVK_LOG_FORMAT_EXCEPTIONS)
		catch (std::exception const& e)
		{
			ASSERT(false, e.what());
		}
#endif
	}
}

void logToFile(std::filesystem::path path, Time pollRate = Time::from_s(0.5f));
void stopFileLogging();
} // namespace le::log
