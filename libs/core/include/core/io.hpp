#pragma once
#include <filesystem>
#include <sstream>
#include <string_view>
#include "utils.hpp"

namespace le
{
namespace stdfs = std::filesystem;

class IOReader
{
protected:
	std::string m_medium;

public:
	IOReader() noexcept;
	IOReader(IOReader&&) noexcept;
	IOReader& operator=(IOReader&&) noexcept;
	IOReader(IOReader const&);
	IOReader& operator=(IOReader const&);
	virtual ~IOReader();

public:
	[[nodiscard]] TResult<std::string> getString(stdfs::path const& id) const;
	[[nodiscard]] bool isPresent(stdfs::path const& id) const;
	[[nodiscard]] bool checkPresence(stdfs::path const& id) const;
	[[nodiscard]] bool checkPresences(std::initializer_list<stdfs::path> ids) const;
	[[nodiscard]] bool checkPresences(ArrayView<stdfs::path const> ids) const;

	std::string_view medium() const;

public:
	[[nodiscard]] virtual bool mount(stdfs::path path) = 0;
	[[nodiscard]] virtual TResult<bytearray> getBytes(stdfs::path const& id) const = 0;
	[[nodiscard]] virtual TResult<std::stringstream> getStr(stdfs::path const& id) const = 0;

protected:
	virtual TResult<stdfs::path> findPrefixed(stdfs::path const& id) const = 0;
};

class FileReader final : public IOReader
{
private:
	std::vector<stdfs::path> m_prefixes;

public:
	static TResult<stdfs::path> findUpwards(stdfs::path const& leaf, std::initializer_list<stdfs::path> anyOf, u8 maxHeight = 10);
	static TResult<stdfs::path> findUpwards(stdfs::path const& leaf, ArrayView<stdfs::path const> anyOf, u8 maxHeight = 10);

public:
	FileReader() noexcept;

public:
	stdfs::path fullPath(stdfs::path const& id) const;

public:
	bool mount(stdfs::path path) override;
	TResult<bytearray> getBytes(stdfs::path const& id) const override;
	TResult<std::stringstream> getStr(stdfs::path const& id) const override;

protected:
	TResult<stdfs::path> findPrefixed(stdfs::path const& id) const override;

private:
	std::vector<stdfs::path> finalPaths(stdfs::path const& id) const;
};

class ZIPReader final : public IOReader
{
public:
	ZIPReader();

public:
	bool mount(stdfs::path path) override;
	TResult<bytearray> getBytes(stdfs::path const& id) const override;
	TResult<std::stringstream> getStr(stdfs::path const& id) const override;

protected:
	TResult<stdfs::path> findPrefixed(stdfs::path const& id) const override;
};

class FileMonitor
{
public:
	enum class Mode : s8
	{
		eTimestamp,
		eTextContents,
		eBinaryContents
	};

	enum class Status : s8
	{
		eUpToDate,
		eNotFound,
		eModified,
		eCOUNT_
	};

protected:
	static FileReader s_reader;

protected:
	stdfs::file_time_type m_lastWriteTime = {};
	stdfs::file_time_type m_lastModifiedTime = {};
	stdfs::path m_path;
	std::string m_text;
	bytearray m_bytes;
	Mode m_mode;
	Status m_status = Status::eNotFound;

public:
	FileMonitor(stdfs::path const& path, Mode mode);
	FileMonitor(FileMonitor&&);
	FileMonitor& operator=(FileMonitor&&);
	virtual ~FileMonitor();

public:
	virtual Status update();

public:
	Status lastStatus() const;
	stdfs::file_time_type lastWriteTime() const;
	stdfs::file_time_type lastModifiedTime() const;

	stdfs::path const& path() const;
	std::string_view text() const;
	bytearray const& bytes() const;
};
} // namespace le
