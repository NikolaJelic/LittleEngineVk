#include <core/utils.hpp>
#include <graphics/context/device.hpp>
#include <graphics/context/memory.hpp>

namespace le::graphics {
vk::SharingMode QShare::operator()(Device const& device, QFlags queues) const {
	return device.m_queues.familyIndices(queues).size() == 1 ? vk::SharingMode::eExclusive : desired;
}

Memory::Memory(Device& device) : m_device(device) {
	VmaAllocatorCreateInfo allocatorInfo = {};
	Instance& inst = device.m_instance;
	auto& dl = inst.m_loader;
	allocatorInfo.instance = static_cast<VkInstance>(inst.m_instance);
	allocatorInfo.device = static_cast<VkDevice>(device.m_device);
	allocatorInfo.physicalDevice = static_cast<VkPhysicalDevice>(device.m_physicalDevice.device);
	VmaVulkanFunctions vkFunc = {};
	vkFunc.vkGetPhysicalDeviceProperties = dl.vkGetPhysicalDeviceProperties;
	vkFunc.vkGetPhysicalDeviceMemoryProperties = dl.vkGetPhysicalDeviceMemoryProperties;
	vkFunc.vkAllocateMemory = dl.vkAllocateMemory;
	vkFunc.vkFreeMemory = dl.vkFreeMemory;
	vkFunc.vkMapMemory = dl.vkMapMemory;
	vkFunc.vkUnmapMemory = dl.vkUnmapMemory;
	vkFunc.vkFlushMappedMemoryRanges = dl.vkFlushMappedMemoryRanges;
	vkFunc.vkInvalidateMappedMemoryRanges = dl.vkInvalidateMappedMemoryRanges;
	vkFunc.vkBindBufferMemory = dl.vkBindBufferMemory;
	vkFunc.vkBindImageMemory = dl.vkBindImageMemory;
	vkFunc.vkGetBufferMemoryRequirements = dl.vkGetBufferMemoryRequirements;
	vkFunc.vkGetImageMemoryRequirements = dl.vkGetImageMemoryRequirements;
	vkFunc.vkCreateBuffer = dl.vkCreateBuffer;
	vkFunc.vkDestroyBuffer = dl.vkDestroyBuffer;
	vkFunc.vkCreateImage = dl.vkCreateImage;
	vkFunc.vkDestroyImage = dl.vkDestroyImage;
	vkFunc.vkCmdCopyBuffer = dl.vkCmdCopyBuffer;
	vkFunc.vkGetBufferMemoryRequirements2KHR = dl.vkGetBufferMemoryRequirements2KHR;
	vkFunc.vkGetImageMemoryRequirements2KHR = dl.vkGetImageMemoryRequirements2KHR;
	vkFunc.vkBindBufferMemory2KHR = dl.vkBindBufferMemory2KHR;
	vkFunc.vkBindImageMemory2KHR = dl.vkBindImageMemory2KHR;
	vkFunc.vkGetPhysicalDeviceMemoryProperties2KHR = dl.vkGetPhysicalDeviceMemoryProperties2KHR;
	allocatorInfo.pVulkanFunctions = &vkFunc;
	vmaCreateAllocator(&allocatorInfo, &m_allocator);
	for (auto& count : m_allocations) {
		count.store(0);
	}
	g_log.log(lvl::info, 1, "[{}] Memory constructed", g_name);
}

Memory::~Memory() {
	{
		auto lock = m_mutex.lock();
		for (auto& buffer : m_buffers) {
			unmapMemory(buffer);
			destroyImpl(buffer, false);
		}
		for (auto& image : m_images) {
			destroyImpl(image);
		}
	}
	vmaDestroyAllocator(m_allocator);
	g_log.log(lvl::info, 1, "[{}] Memory destroyed", g_name);
}

View<Buffer> Memory::construct(Buffer::CreateInfo const& info, [[maybe_unused]] bool bSilent) {
	Buffer buffer;
	Device& d = m_device;
#if defined(LEVK_VKRESOURCE_NAMES)
	ENSURE(!info.name.empty(), "Unnamed buffer!");
	buffer.name = info.name;
#endif
	vk::BufferCreateInfo bufferInfo;
	buffer.writeSize = bufferInfo.size = info.size;
	bufferInfo.usage = info.usage;
	auto const indices = d.m_queues.familyIndices(info.queueFlags);
	bufferInfo.sharingMode = info.share(m_device, info.queueFlags);
	bufferInfo.queueFamilyIndexCount = (u32)indices.size();
	bufferInfo.pQueueFamilyIndices = indices.data();
	VmaAllocationCreateInfo createInfo = {};
	createInfo.usage = info.vmaUsage;
	auto const vkBufferInfo = static_cast<VkBufferCreateInfo>(bufferInfo);
	VkBuffer vkBuffer;
	if (vmaCreateBuffer(m_allocator, &vkBufferInfo, &createInfo, &vkBuffer, &buffer.handle, nullptr) != VK_SUCCESS) {
		throw std::runtime_error("Allocation error");
	}
	buffer.buffer = vk::Buffer(vkBuffer);
	buffer.queueFlags = info.queueFlags;
	buffer.mode = bufferInfo.sharingMode;
	buffer.usage = info.usage;
	buffer.type = info.vmaUsage == VMA_MEMORY_USAGE_GPU_ONLY ? Buffer::Type::eGpuOnly : Buffer::Type::eCpuToGpu;
	VmaAllocationInfo allocationInfo;
	vmaGetAllocationInfo(m_allocator, buffer.handle, &allocationInfo);
	buffer.info = {vk::DeviceMemory(allocationInfo.deviceMemory), allocationInfo.offset, allocationInfo.size};
	m_allocations[(std::size_t)ResourceType::eBuffer].fetch_add(buffer.writeSize);
	if (m_bLogAllocs) {
		if (!bSilent) {
			auto [size, unit] = utils::friendlySize(buffer.writeSize);
#if defined(LEVK_VKRESOURCE_NAMES)
			g_log.log(m_logLevel, 1, "== [{}] Buffer [{}] allocated: [{:.2f}{}] | {}", g_name, buffer.name, size, unit, logCount());
#else
			g_log.log(m_logLevel, 1, "== [{}] Buffer allocated: [{:.2f}{}] | {}", g_name, size, unit, logCount());
#endif
		}
	}
	auto lock = m_mutex.lock();
	buffer.guid = m_buffers.nextID();
	auto const guid = m_buffers.push(std::move(buffer));
	return *m_buffers.find(guid);
}

bool Memory::destroy(View<Buffer> buffer, [[maybe_unused]] bool bSilent) {
	if (!buffer || default_v(buffer->buffer)) {
		return false;
	}
	auto lock = m_mutex.lock();
	if (m_buffers.contains(buffer->guid)) {
		destroyImpl(buffer, bSilent);
		m_buffers.pop(buffer->guid);
		return true;
	}
	return false;
}

bool Memory::mapMemory(Buffer& out_buffer) const {
	if (out_buffer.type != Buffer::Type::eCpuToGpu) {
#if defined(LEVK_VKRESOURCE_NAMES)
		g_log.log(lvl::error, 1, "[{}] Attempt to map GPU-only Buffer [{}]!", g_name, out_buffer.name);
#else
		g_log.log(lvl::error, 1, "[{}] Attempt to map GPU-only Buffer!", g_name);
#endif
		return false;
	}
	if (!out_buffer.pMap && out_buffer.writeSize > 0) {
		vmaMapMemory(m_allocator, out_buffer.handle, &out_buffer.pMap);
		return true;
	}
	return out_buffer.pMap != nullptr;
}

void Memory::unmapMemory(Buffer& out_buffer) const {
	if (out_buffer.pMap) {
		vmaUnmapMemory(m_allocator, out_buffer.handle);
		out_buffer.pMap = nullptr;
	}
}

bool Memory::write(Buffer& out_buffer, void const* pData, Buffer::Span const& range) const {
	if (out_buffer.type != Buffer::Type::eCpuToGpu) {
#if defined(LEVK_VKRESOURCE_NAMES)
		g_log.log(lvl::error, 1, "[{}] Attempt to write to GPU-only Buffer [{}]!", g_name, out_buffer.name);
#else
		g_log.log(lvl::error, 1, "[{}] Attempt to write to GPU-only Buffer!", g_name);
#endif
		return false;
	}
	if (!default_v(out_buffer.info.memory) && !default_v(out_buffer.buffer)) {
		std::size_t const size = range.size == 0 ? (std::size_t)out_buffer.writeSize : range.size;
		if (mapMemory(out_buffer)) {
			void* pStart = (void*)((char*)out_buffer.pMap + range.offset);
			std::memcpy(pStart, pData, size);
		}
		return true;
	}
	return false;
}

View<Image> Memory::construct(Image::CreateInfo const& info) {
	Image ret;
	Device& d = m_device;
#if defined(LEVK_VKRESOURCE_NAMES)
	ENSURE(!info.name.empty(), "Unnamed buffer!");
	ret.name = info.name;
#endif
	vk::ImageCreateInfo imageInfo = info.createInfo;
	auto const indices = d.m_queues.familyIndices(info.queueFlags);
	imageInfo.sharingMode = info.share(d, info.queueFlags);
	imageInfo.queueFamilyIndexCount = (u32)indices.size();
	imageInfo.pQueueFamilyIndices = indices.data();
	VmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = info.vmaUsage;
	auto const vkImageInfo = static_cast<VkImageCreateInfo>(imageInfo);
	VkImage vkImage;
	if (vmaCreateImage(m_allocator, &vkImageInfo, &allocInfo, &vkImage, &ret.handle, nullptr) != VK_SUCCESS) {
		throw std::runtime_error("Allocation error");
	}
	ret.extent = info.createInfo.extent;
	ret.image = vk::Image(vkImage);
	auto const requirements = d.m_device.getImageMemoryRequirements(ret.image);
	ret.queueFlags = info.queueFlags;
	VmaAllocationInfo allocationInfo;
	vmaGetAllocationInfo(m_allocator, ret.handle, &allocationInfo);
	ret.info = {vk::DeviceMemory(allocationInfo.deviceMemory), allocationInfo.offset, allocationInfo.size};
	ret.allocatedSize = requirements.size;
	ret.mode = imageInfo.sharingMode;
	m_allocations[(std::size_t)ResourceType::eImage].fetch_add(ret.allocatedSize);
	if (m_bLogAllocs) {
		auto [size, unit] = utils::friendlySize(ret.allocatedSize);
#if defined(LEVK_VKRESOURCE_NAMES)
		g_log.log(m_logLevel, 1, "== [{}] Image [{}] allocated: [{:.2f}{}] | {}", g_name, ret.name, size, unit, logCount());
#else
		g_log.log(m_logLevel, 1, "== [{}] Image allocated: [{:.2f}{}] | {}", g_name, size, unit, logCount());
#endif
	}
	auto lock = m_mutex.lock();
	ret.guid = m_images.nextID();
	auto const guid = m_images.push(std::move(ret));
	return *m_images.find(guid);
}

bool Memory::destroy(View<Image> image) {
	if (!image || default_v(image->image)) {
		return false;
	}
	auto lock = m_mutex.lock();
	if (m_images.contains(image->guid)) {
		destroyImpl(image);
		m_images.pop(image->guid);
		image = {};
		return true;
	}
	return false;
}

std::string Memory::logCount() {
	auto [bufferSize, bufferUnit] = utils::friendlySize(m_allocations[(std::size_t)ResourceType::eBuffer]);
	auto const [imageSize, imageUnit] = utils::friendlySize(m_allocations[(std::size_t)ResourceType::eImage]);
	return fmt::format("Buffers: [{:.2f}{}]; Images: [{:.2f}{}]", bufferSize, bufferUnit, imageSize, imageUnit);
}

void Memory::destroyImpl(Buffer& out_buffer, bool bSilent) {
	unmapMemory(out_buffer);
	vmaDestroyBuffer(m_allocator, static_cast<VkBuffer>(out_buffer.buffer), out_buffer.handle);
	m_allocations[(std::size_t)ResourceType::eBuffer].fetch_sub(out_buffer.writeSize);
	if (m_bLogAllocs) {
		if (!bSilent) {
			if (out_buffer.info.actualSize) {
				auto [size, unit] = utils::friendlySize(out_buffer.writeSize);
#if defined(LEVK_VKRESOURCE_NAMES)
				g_log.log(m_logLevel, 1, "-- [{}] Buffer [{}] released: [{:.2f}{}] | {}", g_name, out_buffer.name, size, unit, logCount());
#else
				g_log.log(m_logLevel, 1, "-- [{}] Buffer released: [{:.2f}{}] | {}", g_name, size, unit, logCount());
#endif
			}
		}
	}
}
void Memory::destroyImpl(Image& out_image) {
	vmaDestroyImage(m_allocator, static_cast<VkImage>(out_image.image), out_image.handle);
	m_allocations[(std::size_t)ResourceType::eImage].fetch_sub(out_image.allocatedSize);
	if (m_bLogAllocs) {
		if (out_image.info.actualSize > 0) {
			auto [size, unit] = utils::friendlySize(out_image.allocatedSize);
#if defined(LEVK_VKRESOURCE_NAMES)
			g_log.log(m_logLevel, 1, "-- [{}] Image [{}] released: [{:.2f}{}] | {}", g_name, out_image.name, size, unit, logCount());
#else
			g_log.log(m_logLevel, 1, "-- [{}] Image released: [{:.2f}{}] | {}", g_name, size, unit, logCount());
#endif
		}
	}
}
} // namespace le::graphics
