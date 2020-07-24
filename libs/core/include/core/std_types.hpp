#pragma once
#if defined(__MINGW32__)
#define __STDC_FORMAT_MACROS
#endif
#include <array>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <utility>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(near)
#undef near
#endif
#if defined(far)
#undef far
#endif
#if defined(min)
#undef min
#endif
#if defined(max)
#undef max
#endif

namespace le
{
using u8 = std::uint8_t;
using s8 = std::int8_t;
using u16 = std::uint16_t;
using s16 = std::int16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using s32 = std::int32_t;
using s64 = std::int64_t;
using f32 = float;
using f64 = double;
using bytearray = std::vector<std::byte>;

template <typename T, typename E, std::size_t N = (std::size_t)E::eCOUNT_>
using EnumArray = std::array<T, N>;

template <typename... Ts>
constexpr bool alwaysFalse = false;

template <typename... Ts>
constexpr bool alwaysTrue = true;

template <typename T>
struct FalseType final : std::false_type
{
};

template <typename T>
struct TrueType final : std::true_type
{
};

///
/// \brief Structured Binding of a payload and a `bool` (indicating the result of an operation)
///
template <typename T>
struct TResult
{
	using type = T;

	T payload;
	bool bResult = false;

	TResult() = default;
	TResult(T&& payload) : payload(std::forward<T>(payload)), bResult(true) {}
	TResult(T&& payload, bool bResult) : payload(std::forward<T>(payload)), bResult(bResult) {}
};

///
/// \brief Obtain the number of elements in a stack array
///
template <typename T, std::size_t N>
constexpr std::size_t arraySize(T (&)[N])
{
	return N;
}

///
/// \brief Obtain the max value for `T`
///
template <typename T>
constexpr T maxVal()
{
	static_assert(std::is_arithmetic_v<T>, "T must be arithemtic!");
	return std::numeric_limits<T>::max();
}
} // namespace le
