#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <core/ensure.hpp>
#include <core/log.hpp>
#include <core/os.hpp>
#include <core/threads.hpp>
#if defined(LEVK_OS_WINX)
#include <Windows.h>
#elif defined(LEVK_OS_LINUX)
#include <cstring>
#include <iostream>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace le {
namespace stdfs = std::filesystem;

namespace {
stdfs::path g_exeLocation;
stdfs::path g_exePath;
stdfs::path g_workingDir;
std::string g_exePathStr;
std::deque<os::ArgsParser::entry> g_args;
} // namespace

os::Service::Service(os::Args args) {
	if (g_exeLocation.empty() && args.argc > 0) {
		os::args(args);
	}
	threads::init();
}

os::Service::~Service() {
	threads::joinAll();
}

void os::args(Args args) {
	g_workingDir = stdfs::absolute(stdfs::current_path());
	if (args.argc > 0) {
		ArgsParser parser;
		g_args = parser.parse(args.argc, args.argv);
		auto& arg0 = g_args.front();
		g_exeLocation = stdfs::absolute(arg0.k);
		g_exePath = g_exeLocation.parent_path();
		while (g_exePath.filename() == ".") {
			g_exePath = g_exePath.parent_path();
		}
		g_exeLocation = g_exePath / g_exeLocation.filename();
		g_args.pop_front();
	}
}

std::string os::argv0() {
	return g_exeLocation.generic_string();
}

stdfs::path os::dirPath(Dir dir) {
	switch (dir) {
	default:
	case os::Dir::eWorking:
		if (g_workingDir.empty()) {
			g_workingDir = stdfs::absolute(stdfs::current_path());
		}
		return g_workingDir;
	case os::Dir::eExecutable:
		if (g_exePath.empty()) {
			logW("[OS] Unknown executable path! Using working directory instead [{}]", g_workingDir.generic_string());
			g_exePath = dirPath(Dir::eWorking);
		}
		return g_exePath;
	}
}

std::deque<os::ArgsParser::entry> const& os::args() noexcept {
	return g_args;
}

bool os::isDebuggerAttached() {
	bool ret = false;
#if defined(LEVK_OS_WINX)
	ret = IsDebuggerPresent() != 0;
#elif defined(LEVK_OS_LINUX)
	char buf[4096];
	auto const status_fd = ::open("/proc/self/status", O_RDONLY);
	if (status_fd == -1) {
		return false;
	}
	auto const num_read = ::read(status_fd, buf, sizeof(buf) - 1);
	if (num_read <= 0) {
		return false;
	}
	buf[(std::size_t)num_read] = '\0';
	constexpr char tracerPidString[] = "TracerPid:";
	auto const tracer_pid_ptr = ::strstr(buf, tracerPidString);
	if (!tracer_pid_ptr) {
		return false;
	}
	for (char const* pChar = tracer_pid_ptr + sizeof(tracerPidString) - 1; pChar <= buf + num_read && *pChar != '\n'; ++pChar) {
		if (::isspace(*pChar)) {
			continue;
		} else {
			ret = ::isdigit(*pChar) != 0 && *pChar != '0';
		}
	}
#endif
	return ret;
}

void os::debugBreak() {
#if defined(LEVK_RUNTIME_MSVC)
	__debugbreak();
#elif defined(LEVK_RUNTIME_LIBSTDCXX)
#ifdef SIGTRAP
	raise(SIGTRAP);
#endif
#endif
	return;
}

bool os::sysCall(std::string_view command) {
	if (std::system(command.data()) == 0) {
		return true;
	}
	return false;
}
} // namespace le
