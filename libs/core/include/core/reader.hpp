#pragma once
#include <filesystem>
#include <sstream>
#include <string_view>
#include <core/span.hpp>
#include <core/utils.hpp>
#include <kt/result/result.hpp>

namespace le {
namespace stdfs = std::filesystem;

namespace io {
///
/// \brief Abstract base class for reading data from various IO
///
class Reader {
  public:
	template <typename T>
	using Result = kt::result_void<T>;

  public:
	Reader() noexcept;
	Reader(Reader&&) noexcept;
	Reader& operator=(Reader&&) noexcept;
	Reader(Reader const&);
	Reader& operator=(Reader const&);
	virtual ~Reader();

  public:
	///
	/// \brief Check if an `id` is present to load
	///
	[[nodiscard]] bool isPresent(stdfs::path const& id) const;
	///
	/// \brief Check if an `id` is present to load, and log an error if not
	///
	[[nodiscard]] bool checkPresence(stdfs::path const& id) const;
	///
	/// \brief Check if `id`s are present to load, and log errors if not
	///
	[[nodiscard]] bool checkPresences(Span<stdfs::path> ids) const;
	///
	/// \brief Check if `id`s are present to load, and log errors if not
	///
	[[nodiscard]] bool checkPresences(std::initializer_list<stdfs::path> ids) const;
	///
	/// \brief Obtain data as `std::string`
	///
	[[nodiscard]] Result<std::string> string(stdfs::path const& id) const;
	///
	/// \brief Obtain the IO medium (of the concrete class)
	///
	std::string_view medium() const;

  public:
	///
	/// \brief Mount a path on the IO medium
	/// Mounted paths are prefixed to `id`s being searched
	///
	[[nodiscard]] virtual bool mount(stdfs::path path) = 0;
	///
	/// \brief Obtain data as `bytearray` (`std::vector<std::byte>`)
	///
	[[nodiscard]] virtual Result<bytearray> bytes(stdfs::path const& id) const = 0;
	///
	/// \brief Obtain data as `std::stringstream`
	///
	[[nodiscard]] virtual Result<std::stringstream> sstream(stdfs::path const& id) const = 0;

  protected:
	std::string m_medium;

  protected:
	virtual Result<stdfs::path> findPrefixed(stdfs::path const& id) const = 0;
};

///
/// \brief Concrete class for filesystem IO
///
class FileReader final : public Reader {
  public:
	///
	/// \brief Obtain full path to directory containing any of `anyOf` `id`s.
	/// \param leaf directory to start searching upwards from
	/// \param anyOf list of `id`s to search for a match for
	/// \param maxHeight maximum recursive depth
	///
	static Result<stdfs::path> findUpwards(stdfs::path const& leaf, Span<stdfs::path> anyOf, u8 maxHeight = 10);

  public:
	FileReader() noexcept;

  public:
	///
	/// \brief Obtain fully qualified path (if `id` is found)
	///
	stdfs::path fullPath(stdfs::path const& id) const;

  public:
	///
	/// \brief Mount filesystem directory
	///
	bool mount(stdfs::path path) override;
	Result<bytearray> bytes(stdfs::path const& id) const override;
	Result<std::stringstream> sstream(stdfs::path const& id) const override;

  private:
	std::vector<stdfs::path> m_dirs;

  private:
	Result<stdfs::path> findPrefixed(stdfs::path const& id) const override;

  private:
	std::vector<stdfs::path> finalPaths(stdfs::path const& id) const;
};

///
/// \brief Concrete class for `.zip` IO
///
class ZIPReader final : public Reader {
  public:
	ZIPReader();

  public:
	///
	/// \brief Mount `.zip` file
	///
	bool mount(stdfs::path path) override;
	Result<bytearray> bytes(stdfs::path const& id) const override;
	Result<std::stringstream> sstream(stdfs::path const& id) const override;

  private:
	std::vector<stdfs::path> m_zips;

  private:
	Result<stdfs::path> findPrefixed(stdfs::path const& id) const override;
};

///
/// \brief Utility for monitoring filesystem files
///
class FileMonitor {
  public:
	///
	/// \brief Monitoring mode
	///
	enum class Mode : s8 { eTimestamp, eTextContents, eBinaryContents };

	///
	/// \brief Monitor status
	///
	enum class Status : s8 { eUpToDate, eNotFound, eModified, eCOUNT_ };

  public:
	///
	/// \brief Constructor
	/// \param path: fully qualified file path to monitor
	/// \param mode: mode to operate the monitor in
	///
	FileMonitor(stdfs::path const& path, Mode mode);
	FileMonitor(FileMonitor&&);
	FileMonitor& operator=(FileMonitor&&);
	virtual ~FileMonitor();

  public:
	///
	/// \brief Obtain current status of file being monitored
	///
	virtual Status update();

  public:
	///
	/// \brief Obtain previous status of file being monitored
	///
	Status lastStatus() const;
	///
	/// \brief Obtain write-to-file timestamp
	///
	stdfs::file_time_type lastWriteTime() const;
	///
	/// \brief Obtain last modified timestamp
	///
	stdfs::file_time_type lastModifiedTime() const;

	///
	/// \brief Obtain the file path being monitored
	///
	stdfs::path const& path() const;
	///
	/// \brief Obtain the last scanned contents of the file being monitored
	/// Note: only valid for `eTextContents` mode
	///
	std::string_view text() const;
	///
	/// \brief Obtain the last scanned contents of the file being monitored
	/// Note: only valid for `eBinaryContents` mode
	bytearray const& bytes() const;

  protected:
	inline static io::FileReader s_reader;

  protected:
	stdfs::file_time_type m_lastWriteTime = {};
	stdfs::file_time_type m_lastModifiedTime = {};
	stdfs::path m_path;
	std::string m_text;
	bytearray m_bytes;
	Mode m_mode;
	Status m_status = Status::eNotFound;
};
} // namespace io
} // namespace le
