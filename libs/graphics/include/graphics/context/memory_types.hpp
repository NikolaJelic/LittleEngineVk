#pragma once
#include <vk_mem_alloc.h>
#include <core/std_types.hpp>
#include <graphics/qflags.hpp>
#include <vulkan/vulkan.hpp>

#if defined(LEVK_DEBUG)
#if !defined(LEVK_VKRESOURCE_NAMES)
#define LEVK_VKRESOURCE_NAMES
#endif
#endif

namespace le::graphics {
struct AllocInfo final {
	vk::DeviceMemory memory;
	vk::DeviceSize offset = {};
	vk::DeviceSize actualSize = {};
};

struct VkResource {
#if defined(LEVK_VKRESOURCE_NAMES)
	std::string name;
#endif
	AllocInfo info;
	VmaAllocation handle;
	QFlags queueFlags;
	vk::SharingMode mode;
	u64 guid = 0;
};

struct Buffer final : VkResource {
	enum class Type { eCpuToGpu, eGpuOnly };
	struct CreateInfo;
	struct Span;

	vk::Buffer buffer;
	vk::DeviceSize writeSize = {};
	vk::BufferUsageFlags usage;
	Type type;
	void* pMap = nullptr;
};

struct Buffer::Span {
	std::size_t offset = 0;
	std::size_t size = 0;
};

struct Image final : VkResource {
	struct CreateInfo;

	vk::Image image;
	vk::DeviceSize allocatedSize = {};
	vk::Extent3D extent = {};
	u32 layerCount = 1;
};
} // namespace le::graphics
