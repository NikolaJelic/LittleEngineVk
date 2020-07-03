#include <array>
#include <sstream>
#include <utility>
#include <core/log.hpp>
#include <core/gdata.hpp>
#include <core/utils.hpp>

namespace le
{
namespace
{
struct Escape final
{
	struct Sequence final
	{
		std::pair<char, char> match;
		s64 count = 0;
	};
	std::vector<Sequence> sequences;

	s64 stackSize(char c);
	void add(std::pair<char, char> match);
};

std::string const g_tName = utils::tName<GData>();

std::string sanitise(std::string_view str, std::size_t start, std::size_t end);
bool isWhitespace(char c, u64* out_pLine = nullptr);
bool isBoolean(std::string_view str, std::size_t first);

s64 Escape::stackSize(char c)
{
	s64 total = 0;
	for (auto& sequence : sequences)
	{
		if (c == sequence.match.first && sequence.match.first == sequence.match.second)
		{
			sequence.count = (sequence.count == 0) ? 1 : 0;
		}
		else
		{
			if (c == sequence.match.second)
			{
				ASSERT(sequence.count > 0, "Invalid escape sequence count!");
				--sequence.count;
			}
			else if (c == sequence.match.first)
			{
				++sequence.count;
			}
		}
		total += sequence.count;
	}
	return total;
}

void Escape::add(std::pair<char, char> match)
{
	sequences.push_back({match, 0});
}

std::string sanitise(std::string_view str, std::size_t begin = 0, std::size_t end = 0)
{
	std::string ret;
	ret.reserve(end - begin);
	bool bEscaped = false;
	if (end == 0)
	{
		end = str.size();
	}
	for (std::size_t idx = begin; idx < end; ++idx)
	{
		if (idx > 0 && str.at(idx) == '\\' && str.at(idx - 1) == '\\')
		{
			bEscaped = !bEscaped;
		}
		if (str.at(idx) != '\\' || bEscaped)
		{
			ret += str.at(idx);
		}
	}
	return ret;
}

bool isWhitespace(char c, u64* out_pLine)
{
	if (c == '\n' || c == '\r')
	{
		if (out_pLine)
		{
			++*out_pLine;
		}
		return true;
	}
	return c == ' ' || c == '\t';
}

bool isBoolean(std::string_view str, std::size_t begin)
{
	static std::array<std::string_view, 2> const s_valid = {"true", "false"};
	if (begin < str.size())
	{
		for (auto const& valid : s_valid)
		{
			std::size_t const end = begin + valid.size();
			if (end < str.size() && std::string_view(str.data() + begin, end - begin) == valid)
			{
				return true;
			}
		}
	}
	return false;
}
} // namespace

bool GData::read(std::string json)
{
	m_raw = std::move(json);
	bool bStarted = false;
	u64 line = 1;
	for (std::size_t idx = 0; idx < m_raw.size();)
	{
		advance(idx, line);
		if (idx >= m_raw.size())
		{
			break;
		}
		ASSERT(idx < m_raw.size(), "Invariant violated!");
		if (bStarted)
		{
			while (idx < m_raw.size() && (m_raw.at(idx) == '}' || isWhitespace(m_raw.at(idx), &line)))
			{
				++idx;
			}
		}
		else
		{
			if (m_raw.at(idx) != '{')
			{
				LOG_E("[{}] Expected '{' at index [{}] (line: {})", g_tName, idx, line);
				return false;
			}
			++idx;
		}
		bStarted = true;
		auto key = parseKey(idx, line);
		if (key.empty())
		{
			return false;
		}
		auto const [begin, end] = parseValue(idx, line);
		if (end <= begin)
		{
			return false;
		}
		if (m_fields.find(key) != m_fields.end())
		{
			LOG_W("[{}] Duplicate key [{}] at index [{}] (line: {})! Overwriting value...", g_tName, key, idx, line);
		}
		m_fields[key] = {begin, end};
	}
	if (m_fields.empty())
	{
		LOG_W("[{}] Empty json / nothing parsed", g_tName);
		return false;
	}
	return true;
}

std::string GData::getString(std::string const& key) const
{
	if (auto search = m_fields.find(key); search != m_fields.end())
	{
		auto const [begin, end] = search->second;
		return sanitise(std::string_view(m_raw.data() + begin, end - begin));
	}
	return {};
}

std::vector<std::string> GData::getArray(std::string const& key) const
{
	std::vector<std::string> ret;
	if (auto search = m_fields.find(key); search != m_fields.end())
	{
		auto const [begin, end] = search->second;
		std::string_view value(m_raw.data() + begin, end - begin);
		if (value.size() > 2 && value.at(0) == '[' && value.at(value.size() - 1) == ']')
		{
			std::size_t idx = 1;
			Escape escape;
			escape.add({'[', ']'});
			escape.add({'{', '}'});
			escape.stackSize('[');
			while (idx < value.size())
			{
				while (idx < value.size() && isWhitespace(value.at(idx)))
				{
					++idx;
				}
				std::size_t first = idx;
				while (idx < value.size())
				{
					s64 const stack = escape.stackSize(value.at(idx));
					bool const bNext = stack <= 1 && value.at(idx) == ',';
					bool const bEnd = stack == 0 && value.at(idx) == ']';
					if (bNext || bEnd)
					{
						break;
					}
					++idx;
				}
				std::size_t last = idx >= value.size() ? value.size() - 1 : idx;
				if (value.at(last) == ']')
				{
					--last;
				}
				while (last > first && (value.at(last) == ',' || isWhitespace(value.at(last))))
				{
					--last;
				}
				if (last - first > 0)
				{
					if (value.at(first) == '\"')
					{
						ASSERT(value.at(last) == '\"', "Missing end quote!");
						++first;
						--last;
					}
					ret.push_back(sanitise(value, first, last + 1));
				}
				++idx;
			}
		}
	}
	return ret;
}

std::vector<GData> GData::getDataArray(std::string const& key) const
{
	std::vector<GData> ret;
	auto array = getArray(key);
	for (auto& str : array)
	{
		GData data;
		if (data.read(std::move(str)))
		{
			ret.push_back(std::move(data));
		}
	}
	return ret;
}

GData GData::getData(std::string const& key) const
{
	GData ret;
	if (auto search = m_fields.find(key); search != m_fields.end())
	{
		auto const [begin, end] = search->second;
		if (!ret.read(std::string(m_raw.data() + begin, end - begin)))
		{
			ret.clear();
		}
	}
	return ret;
}

s32 GData::getS32(std::string const& key) const
{
	s32 ret = 0;
	if (auto search = m_fields.find(key); search != m_fields.end())
	{
		auto const [begin, end] = search->second;
		try
		{
			std::string_view value(m_raw.data() + begin, end - begin);
			std::array<char, 128> buffer;
			std::memcpy(buffer.data(), value.data(), value.size());
			buffer.at(value.size()) = '\0';
			ret = (s32)std::atoi(buffer.data());
		}
		catch (const std::exception& e)
		{
			LOG_E("[{}] Failed to parse [{}] into f32! {}", e.what());
		}
	}
	return ret;
}

f64 GData::getF64(std::string const& key) const
{
	f64 ret = 0;
	if (auto search = m_fields.find(key); search != m_fields.end())
	{
		auto const [begin, end] = search->second;
		try
		{
			std::string_view value(m_raw.data() + begin, end - begin);
			std::array<char, 128> buffer;
			std::memcpy(buffer.data(), value.data(), value.size());
			buffer.at(value.size()) = '\0';
			ret = (f64)std::atof(buffer.data());
		}
		catch (const std::exception& e)
		{
			LOG_E("[{}] Failed to parse [{}] into f32! {}", e.what());
		}
	}
	return ret;
}

bool GData::getBool(std::string const& key) const
{
	bool bRet = false;
	if (auto search = m_fields.find(key); search != m_fields.end())
	{
		auto const [begin, end] = search->second;
		std::string_view value(m_raw.data() + begin, end - begin);
		if (value == "1" || value == "true" || value == "TRUE")
		{
			bRet = true;
		}
	}
	return bRet;
}

bool GData::contains(std::string const& key) const
{
	return m_fields.find(key) != m_fields.end();
}

void GData::clear()
{
	m_fields.clear();
	m_raw.clear();
}

std::size_t GData::fieldCount() const
{
	return m_fields.size();
}

std::unordered_map<std::string, std::string> GData::allFields() const
{
	std::unordered_map<std::string, std::string> ret;
	for (auto const& [key, indices] : m_fields)
	{
		ret[key] = std::string(m_raw.data() + indices.first, indices.second - indices.first);
	}
	return ret;
}

std::string GData::parseKey(std::size_t& out_idx, u64& out_line)
{
	static std::string_view const s_failure = "failed to extract key!";
	if (out_idx >= m_raw.size())
	{
		LOG_E("[{}] Unexpected end of string at index [{}] (line: {}), {}", g_tName, out_idx, out_line, s_failure);
		return {};
	}
	advance(out_idx, out_line);
	char c = m_raw.at(out_idx);
	if (c != '\"')
	{
		LOG_E("[{}] Expected: '\"' at index [{}] (line: {}), {}!", g_tName, out_idx, out_line, s_failure);
		return {};
	}
	++out_idx;
	if (out_idx < m_raw.size() && m_raw.at(out_idx) == '\\')
	{
		++out_idx;
	}
	std::size_t const start = out_idx;
	++out_idx;
	if (out_idx >= m_raw.size())
	{
		LOG_E("[{}] Unexpected end of string at index [{}] (line: {}), {}", g_tName, out_idx, out_line, s_failure);
		return {};
	}
	while (out_idx < m_raw.size() && m_raw.at(out_idx) != '"')
	{
		++out_idx;
	}
	std::string ret = sanitise(m_raw, start, out_idx);
	++out_idx;
	advance(out_idx, out_line);
	if (out_idx >= m_raw.size() || m_raw.at(out_idx) != ':')
	{
		LOG_E("[{}] Expected ':' after key [{}] at index [{}] (line: {}), {}", g_tName, ret, out_idx, out_line, s_failure);
		return {};
	}
	++out_idx;
	advance(out_idx, out_line);
	return ret;
}

std::pair<std::size_t, std::size_t> GData::parseValue(std::size_t& out_idx, u64& out_line)
{
	static std::string_view const s_failure = "failed to extract value!";
	if (out_idx >= m_raw.size())
	{
		LOG_E("[{}] Unexpected end of string at index [{}] (line: {}), {}", g_tName, out_idx, out_line, s_failure);
		return {};
	}
	advance(out_idx, out_line);
	char const c = m_raw.at(out_idx);
	bool const bQuoted = c == '\"';
	bool const bArray = !bQuoted && c == '[';
	bool const bObject = !bQuoted && !bArray && c == '{';
	bool const bBoolean = !bQuoted && !bArray && !bObject && isBoolean(m_raw, out_idx);
	bool const bNumeric = !bQuoted && !bArray && !bObject && !bBoolean;
	Escape escape;
	if (bQuoted)
	{
		escape.add({'\"', '\"'});
	}
	else if (bArray)
	{
		escape.add({'[', ']'});
	}
	else if (bObject)
	{
		escape.add({'{', '}'});
	}
	auto isEnd = [&](bool bNoStack = false) -> bool {
		char const x = m_raw.at(out_idx);
		s64 stack = 0;
		if (!bNoStack)
		{
			stack = escape.stackSize(x);
		}
		return stack == 0 && ((bQuoted && x == '\"') || (bArray && x == ']') || x == ',' || x == '}' || (bBoolean && isWhitespace(x, &out_line)));
	};
	if (bQuoted)
	{
		++out_idx;
		escape.stackSize('\"');
	}
	if (out_idx >= m_raw.size())
	{
		LOG_E("[{}] Unexpected end of string at index [{}] (line: {}), {}", g_tName, out_idx, out_line, s_failure);
		return {};
	}
	advance(out_idx, out_line);
	std::size_t const begin = out_idx;
	bool const bNegativeOrNumeric = begin < m_raw.size() && (std::isdigit(m_raw.at(begin)) || m_raw.at(begin) == '-');
	while (out_idx < m_raw.size() && !isEnd())
	{
		if (!bQuoted)
		{
			if (bNumeric && !std::isdigit(m_raw.at(out_idx)) && m_raw.at(out_idx) != '.' && !bNegativeOrNumeric)
			{
				LOG_E("[{}] Expected numeric value at index [{}] (line: {}), {}", g_tName, out_idx, out_line, s_failure);
				return {};
			}
		}
		isWhitespace(m_raw.at(out_idx), &out_line);
		++out_idx;
	}
	if (out_idx >= m_raw.size() || !isEnd(true))
	{
		char e = bQuoted ? '\"' : bArray ? ']' : '}';
		LOG_E("[{}] Expected '{}' at index [{}] (line: {}), {}", g_tName, e, out_idx, out_line, s_failure);
		return {};
	}
	if (bArray || bObject)
	{
		++out_idx;
	}
	std::size_t const end = out_idx;
	if (bQuoted)
	{
		++out_idx;
	}
	advance(out_idx, out_line);
	if (out_idx >= m_raw.size() || (m_raw.at(out_idx) != ',' && m_raw.at(out_idx) != '}'))
	{
		LOG_E("[{}] Unterminated value at index [{}] (line: {}), {}", g_tName, out_idx, out_line, s_failure);
		return {};
	}
	++out_idx;
	advance(out_idx, out_line);
	return {begin, end};
}

void GData::advance(std::size_t& out_idx, std::size_t& out_line) const
{
	while (out_idx < m_raw.size() && isWhitespace(m_raw.at(out_idx), &out_line))
	{
		++out_idx;
	}
}
} // namespace le
