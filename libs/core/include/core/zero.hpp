#pragma once
#include <functional>
#include <utility>

namespace le {
///
/// \brief Struct to encapsulate zero-initialised `T` (primitive type)
///
template <typename T, T Zero = 0>
struct TZero final {
	using type = T;

	static constexpr T null = Zero;

	T payload;

	constexpr TZero(T payload = null) noexcept;
	constexpr TZero(TZero&& rhs) noexcept;
	constexpr TZero& operator=(TZero&& rhs) noexcept;
	constexpr TZero(TZero const&) noexcept = default;
	constexpr TZero& operator=(TZero const&) noexcept = default;
	~TZero() = default;

	constexpr operator T() const noexcept;
};

template <typename T, T Zero = 0>
constexpr bool operator==(TZero<T, Zero> lhs, TZero<T, Zero> rhs) {
	return lhs.payload == rhs.payload;
}

template <typename T, T Zero = 0>
constexpr bool operator!=(TZero<T, Zero> lhs, TZero<T, Zero> rhs) {
	return lhs.payload != rhs.payload;
}

template <typename T, T Zero>
constexpr TZero<T, Zero>::TZero(T payload) noexcept : payload(payload) {
}

template <typename T, T Zero>
constexpr TZero<T, Zero>::TZero(TZero<T, Zero>&& rhs) noexcept : payload(rhs.payload) {
	rhs.payload = Zero;
}

template <typename T, T Zero>
constexpr TZero<T, Zero>& TZero<T, Zero>::operator=(TZero&& rhs) noexcept {
	if (&rhs != this) {
		payload = rhs.payload;
		rhs.payload = Zero;
	}
	return *this;
}

template <typename T, T Zero>
constexpr TZero<T, Zero>::operator T() const noexcept {
	return payload;
}
} // namespace le

namespace std {
template <typename T, T Zero>
struct hash<le::TZero<T, Zero>> {
	size_t operator()(le::TZero<T, Zero> zero) const noexcept {
		return std::hash<T>()(zero.payload);
	}
};
} // namespace std
