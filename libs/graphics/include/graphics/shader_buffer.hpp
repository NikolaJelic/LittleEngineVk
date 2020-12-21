#pragma once
#include <graphics/context/device.hpp>
#include <graphics/context/vram.hpp>
#include <graphics/descriptor_set.hpp>
#include <graphics/utils/ring_buffer.hpp>

namespace le::graphics {
struct ShaderBufInfo {
	vk::DescriptorType type = vk::DescriptorType::eUniformBuffer;
	u32 rotateCount = 2;
};

namespace detail {
constexpr vk::BufferUsageFlagBits shaderBufUsage(vk::DescriptorType type) noexcept;
template <typename T, bool IsArray>
struct ShaderBufTraits;
} // namespace detail

template <typename T, bool IsArray>
class ShaderBuffer;

template <typename T, bool IsArray>
using TBuf = ShaderBuffer<T, IsArray>;

template <typename T, bool IsArray>
class ShaderBuffer {
  public:
	using traits = detail::ShaderBufTraits<T, IsArray>;
	using type = typename traits::type;
	using value_type = typename traits::value_type;

	static_assert(std::is_trivial_v<value_type>, "value_type must be trivial");

	static constexpr std::size_t bufSize = sizeof(value_type);

	ShaderBuffer(VRAM& vram, std::string_view name, ShaderBufInfo const& info) : m_vram(vram) {
		m_storage.name = std::string(name);
		m_storage.usage = detail::shaderBufUsage(info.type);
		m_storage.rotateCount = info.rotateCount;
	}

	ShaderBuffer(ShaderBuffer&&) = default;
	ShaderBuffer& operator=(ShaderBuffer&& rhs) {
		if (&rhs != this) {
			destroy();
			m_storage = std::move(rhs.m_storage);
			rhs.m_storage = {};
			m_vram = rhs.m_vram;
		}
		return *this;
	}
	~ShaderBuffer() {
		destroy();
	}

	void set(type const& t) {
		m_storage.t = t;
		write();
	}
	void set(type&& t) {
		m_storage.t = std::move(t);
		write();
	}
	type const& get() const {
		return m_storage.t;
	}
	type& get() {
		return m_storage.t;
	}

	void write(std::optional<T> t = std::nullopt) {
		if (t) {
			m_storage.t = std::move(*t);
		}
		if constexpr (IsArray) {
			resize(m_storage.t.size());
			std::size_t idx = 0;
			for (auto const& t : m_storage.t) {
				m_vram.get().write(m_storage.buffers[idx++].get(), &t, {0, bufSize});
			}
		} else {
			resize(1);
			m_vram.get().write(m_storage.buffers.front().get(), &m_storage.t, {0, bufSize});
		}
	}

	void update(DescriptorSet& out_set, u32 binding) {
		if constexpr (IsArray) {
			resize(m_storage.t.size());
			std::vector<CView<Buffer>> vec;
			vec.reserve(m_storage.t.size());
			for (std::size_t idx = 0; idx < m_storage.t.size(); ++idx) {
				RingBuffer<View<Buffer>> const& rb = m_storage.buffers[idx];
				vec.push_back(rb.get());
			}
			out_set.updateBuffers(binding, vec, bufSize);
		} else {
			resize(1);
			out_set.updateBuffers(binding, CView<Buffer>(m_storage.buffers.front().get()), bufSize);
		}
	}

	void swap() {
		for (auto& rb : m_storage.buffers) {
			rb.next();
		}
	}

  private:
	void resize(std::size_t size) {
		m_storage.buffers.reserve(size);
		for (std::size_t i = m_storage.buffers.size(); i < size; ++i) {
			RingBuffer<View<Buffer>> buffer;
			io::Path prefix(m_storage.name);
			if constexpr (IsArray) {
				prefix += "[";
				prefix += std::to_string(i);
				prefix += "]";
			}
			for (u32 j = 0; j < m_storage.rotateCount; ++j) {
				buffer.ts.push_back(m_vram.get().createBO((prefix / std::to_string(j)).generic_string(), bufSize, m_storage.usage, true));
			}
			m_storage.buffers.push_back(std::move(buffer));
		}
	}

	void destroy() {
		VRAM& v = m_vram;
		v.m_device.get().defer([&v, b = std::move(m_storage.buffers)]() {
			for (RingBuffer<View<Buffer>> const& rb : b) {
				for (View<Buffer> const& buf : rb.ts) {
					v.destroy(buf);
				}
			}
		});
		m_storage = {};
	}

	struct Storage {
		std::vector<RingBuffer<View<Buffer>>> buffers;
		type t;
		std::string name;
		vk::BufferUsageFlagBits usage = {};
		u32 rotateCount = 0;
	};

	Storage m_storage;
	Ref<VRAM> m_vram;
};

// impl

namespace detail {
constexpr vk::BufferUsageFlagBits shaderBufUsage(vk::DescriptorType type) noexcept {
	switch (type) {
	case vk::DescriptorType::eStorageBuffer:
		return vk::BufferUsageFlagBits::eStorageBuffer;
	default:
		return vk::BufferUsageFlagBits::eUniformBuffer;
	}
}

template <typename T, bool IsArray>
struct ShaderBufTraits;

template <typename T>
struct ShaderBufTraits<T, true> {
	using type = T;
	using value_type = typename T::value_type;
};
template <typename T>
struct ShaderBufTraits<T, false> {
	using type = T;
	using value_type = T;
};
} // namespace detail
} // namespace le::graphics
