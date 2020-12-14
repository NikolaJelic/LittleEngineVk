#include <algorithm>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <core/maths.hpp>
#include <graphics/context/device.hpp>
#include <graphics/utils/utils.hpp>

namespace le::graphics {
namespace {
void listDevices(Span<AvailableDevice> devices) {
	std::stringstream str;
	str << "\nAvailable GPUs:";
	std::size_t idx = 0;
	for (auto const& device : devices) {
		str << " [" << idx << "] " << device.name() << '\t';
	}
	str << "\n\n";
	std::cout << str.str();
}

template <typename T, typename U>
T const* fromNextChain(U* pNext, vk::StructureType type) {
	if (pNext) {
		auto* pIn = reinterpret_cast<vk::BaseInStructure const*>(pNext);
		if (pIn->sType == type) {
			return reinterpret_cast<T const*>(pIn);
		}
		return fromNextChain<T>(pIn->pNext, type);
	}
	return nullptr;
}
} // namespace

// Prevent validation spam on Windows
extern dl::level g_validationLevel;

Device::Device(Instance& instance, vk::SurfaceKHR surface, CreateInfo const& info) : m_instance(instance) {
	if (default_v(instance.m_instance)) {
		throw std::runtime_error("Invalid graphics Instance");
	}
	if (default_v(surface)) {
		throw std::runtime_error("Invalid Vulkan surface");
	}
	// Prevent validation spam on Windows
	auto const validationLevel = std::exchange(g_validationLevel, dl::level::warning);
	m_metadata.surface = surface;
	m_metadata.available = availableDevices();
	if (info.bPrintAvailable) {
		listDevices(m_metadata.available);
	}
	if (info.pickDevice) {
		m_metadata.picked = info.pickDevice(m_metadata.available);
		if (!default_v(m_metadata.picked.physicalDevice)) {
			logI("[{}] Using custom GPU: {}", g_name, m_metadata.picked.name());
		}
	}
	if (default_v(m_metadata.picked.physicalDevice) && !m_metadata.available.empty()) {
		for (auto const& availableDevice : m_metadata.available) {
			if (availableDevice.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
				m_metadata.picked = availableDevice;
				break;
			}
		}
		if (default_v(m_metadata.picked.physicalDevice)) {
			m_metadata.picked = m_metadata.available.front();
		}
	}
	if (default_v(m_metadata.picked.physicalDevice)) {
		throw std::runtime_error("Failed to select a physical device!");
	}
	m_physicalDevice = m_metadata.picked.physicalDevice;
	m_metadata.limits = m_metadata.picked.properties.limits;
	m_metadata.lineWidth.first = m_metadata.picked.properties.limits.lineWidthRange[0U];
	m_metadata.lineWidth.second = m_metadata.picked.properties.limits.lineWidthRange[1U];
	auto families = utils::queueFamilies(m_metadata.picked, m_metadata.surface);
	if (info.qselect == QSelect::eSingleFamily || info.qselect == QSelect::eSingleQueue) {
		std::optional<QueueFamily> uber;
		for (auto const& family : families) {
			if (family.flags.all(QFlags::inverse())) {
				uber = family;
				logI("[{}] Forcing single Vulkan queue family [{}]", g_name, family.familyIndex);
				break;
			}
		}
		if (uber) {
			if (info.qselect == QSelect::eSingleQueue) {
				logI("[{}] Forcing single Vulkan queue (family supports [{}])", g_name, uber->total);
				uber->total = 1;
			}
			families = {*uber};
		}
	}
	auto queueCreateInfos = m_queues.select(families);
	vk::PhysicalDeviceFeatures deviceFeatures;
	deviceFeatures.fillModeNonSolid = m_metadata.picked.features2.features.fillModeNonSolid;
	deviceFeatures.wideLines = m_metadata.picked.features2.features.wideLines;
	using DIF = vk::PhysicalDeviceDescriptorIndexingFeatures;
	DIF descriptorIndexingFeatures;
	if (auto pIndexingFeatures = fromNextChain<DIF>(m_metadata.picked.features2.pNext, vk::StructureType::ePhysicalDeviceDescriptorIndexingFeatures)) {
		// TODO: check before enabling
		descriptorIndexingFeatures.runtimeDescriptorArray = pIndexingFeatures->runtimeDescriptorArray;
	}
	vk::DeviceCreateInfo deviceCreateInfo;
	deviceCreateInfo.queueCreateInfoCount = (u32)queueCreateInfos.size();
	deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
	deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
	// deviceCreateInfo.pNext = &descriptorIndexingFeatures;
	if (!instance.m_metadata.layers.empty()) {
		deviceCreateInfo.enabledLayerCount = (u32)instance.m_metadata.layers.size();
		deviceCreateInfo.ppEnabledLayerNames = instance.m_metadata.layers.data();
	}
	deviceCreateInfo.enabledExtensionCount = (u32)requiredExtensions.size();
	deviceCreateInfo.ppEnabledExtensionNames = requiredExtensions.data();
	m_device = m_physicalDevice.createDevice(deviceCreateInfo);
	m_queues.setup(m_device);
	instance.m_loader.init(m_device);
	logD("[{}] Vulkan device constructed, using GPU {}", g_name, m_metadata.picked.name());
	g_validationLevel = validationLevel;
}

std::vector<AvailableDevice> Device::availableDevices() const {
	std::vector<AvailableDevice> ret;
	auto physicalDevices = m_instance.get().m_instance.enumeratePhysicalDevices();
	ret.reserve(physicalDevices.size());
	for (auto const& physDev : physicalDevices) {
		std::unordered_set<std::string_view> missingExtensions(requiredExtensions.begin(), requiredExtensions.end());
		auto const extensions = physDev.enumerateDeviceExtensionProperties();
		for (std::size_t idx = 0; idx < extensions.size() && !missingExtensions.empty(); ++idx) {
			missingExtensions.erase(std::string_view(extensions[idx].extensionName));
		}
		if (missingExtensions.empty()) {
			AvailableDevice availableDevice;
			availableDevice.properties = physDev.getProperties();
			availableDevice.queueFamilies = physDev.getQueueFamilyProperties();
			availableDevice.features2 = physDev.getFeatures2();
			availableDevice.physicalDevice = physDev;
			ret.push_back(std::move(availableDevice));
		}
	}
	return ret;
}

Device::~Device() {
	waitIdle();
	logD_if(!default_v(m_device), "[{}] Vulkan device destroyed", g_name);
	destroy(m_metadata.surface, m_device);
}

bool Device::valid(vk::SurfaceKHR surface) const {
	if (!default_v(m_physicalDevice)) {
		return m_physicalDevice.getSurfaceSupportKHR(m_queues.familyIndex(QType::ePresent), surface);
	}
	return false;
}

void Device::waitIdle() {
	if (!default_v(m_device)) {
		m_device.waitIdle();
	}
	m_deferred.flush();
}

vk::Semaphore Device::createSemaphore() const {
	return m_device.createSemaphore({});
}

vk::Fence Device::createFence(bool bSignalled) const {
	vk::FenceCreateFlags flags = bSignalled ? vk::FenceCreateFlagBits::eSignaled : vk::FenceCreateFlags();
	return m_device.createFence(flags);
}

void Device::resetOrCreateFence(vk::Fence& out_fence, bool bSignalled) const {
	if (default_v(out_fence)) {
		out_fence = createFence(bSignalled);
	} else {
		resetFence(out_fence);
	}
}

void Device::waitFor(vk::Fence optional) const {
	if (!default_v(optional)) {
		if constexpr (levk_debug) {
			static constexpr u64 s_wait = 1000ULL * 1000 * 5000;
			auto const result = m_device.waitForFences(optional, true, s_wait);
			ENSURE(result != vk::Result::eTimeout && result != vk::Result::eErrorDeviceLost, "Fence wait failure!");
			if (result == vk::Result::eTimeout || result == vk::Result::eErrorDeviceLost) {
				logE("[{}] Fence wait failure!", g_name);
			}
		} else {
			m_device.waitForFences(optional, true, maths::max<u64>());
		}
	}
}

void Device::waitAll(vAP<vk::Fence> validFences) const {
	if (!validFences.empty()) {
		if constexpr (levk_debug) {
			static constexpr u64 s_wait = 1000ULL * 1000 * 5000;
			auto const result = m_device.waitForFences(std::move(validFences), true, s_wait);
			ENSURE(result != vk::Result::eTimeout && result != vk::Result::eErrorDeviceLost, "Fence wait failure!");
			if (result == vk::Result::eTimeout || result == vk::Result::eErrorDeviceLost) {
				logE("[{}] Fence wait failure!", g_name);
			}
		} else {
			m_device.waitForFences(std::move(validFences), true, maths::max<u64>());
		}
	}
}

void Device::resetFence(vk::Fence optional) const {
	if (!default_v(optional)) {
		m_device.resetFences(optional);
	}
}

void Device::resetAll(vAP<vk::Fence> validFences) const {
	if (!validFences.empty()) {
		m_device.resetFences(std::move(validFences));
	}
}

bool Device::signalled(Span<vk::Fence> fences) const {
	auto const s = [this](vk::Fence const& fence) -> bool { return default_v(fence) || m_device.getFenceStatus(fence) == vk::Result::eSuccess; };
	return std::all_of(fences.begin(), fences.end(), s);
}

vk::ImageView Device::createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags, vk::ImageViewType type) const {
	vk::ImageViewCreateInfo createInfo;
	createInfo.image = image;
	createInfo.viewType = type;
	createInfo.format = format;
	createInfo.components.r = createInfo.components.g = createInfo.components.b = createInfo.components.a = vk::ComponentSwizzle::eIdentity;
	createInfo.subresourceRange.aspectMask = aspectFlags;
	createInfo.subresourceRange.baseMipLevel = 0;
	createInfo.subresourceRange.levelCount = 1;
	createInfo.subresourceRange.baseArrayLayer = 0;
	createInfo.subresourceRange.layerCount = type == vk::ImageViewType::eCube ? 6 : 1;
	return m_device.createImageView(createInfo);
}

vk::PipelineLayout Device::createPipelineLayout(vAP<vk::PushConstantRange> pushConstants, vAP<vk::DescriptorSetLayout> setLayouts) const {
	vk::PipelineLayoutCreateInfo createInfo;
	createInfo.setLayoutCount = setLayouts.size();
	createInfo.pSetLayouts = setLayouts.data();
	createInfo.pushConstantRangeCount = pushConstants.size();
	createInfo.pPushConstantRanges = pushConstants.data();
	return m_device.createPipelineLayout(createInfo);
}

vk::DescriptorSetLayout Device::createDescriptorSetLayout(vAP<vk::DescriptorSetLayoutBinding> bindings) const {
	vk::DescriptorSetLayoutCreateInfo createInfo;
	createInfo.bindingCount = bindings.size();
	createInfo.pBindings = bindings.data();
	return m_device.createDescriptorSetLayout(createInfo);
}

vk::DescriptorPool Device::createDescriptorPool(vAP<vk::DescriptorPoolSize> poolSizes, u32 maxSets) const {
	vk::DescriptorPoolCreateInfo createInfo;
	createInfo.poolSizeCount = poolSizes.size();
	createInfo.pPoolSizes = poolSizes.data();
	createInfo.maxSets = maxSets;
	return m_device.createDescriptorPool(createInfo);
}

std::vector<vk::DescriptorSet> Device::allocateDescriptorSets(vk::DescriptorPool pool, vAP<vk::DescriptorSetLayout> layouts, u32 setCount) const {
	vk::DescriptorSetAllocateInfo allocInfo;
	allocInfo.descriptorPool = pool;
	allocInfo.descriptorSetCount = setCount;
	allocInfo.pSetLayouts = layouts.data();
	return m_device.allocateDescriptorSets(allocInfo);
}

vk::RenderPass Device::createRenderPass(vAP<vk::AttachmentDescription> attachments, vAP<vk::SubpassDescription> subpasses,
										vAP<vk::SubpassDependency> dependencies) const {
	vk::RenderPassCreateInfo createInfo;
	createInfo.attachmentCount = attachments.size();
	createInfo.pAttachments = attachments.data();
	createInfo.subpassCount = subpasses.size();
	createInfo.pSubpasses = subpasses.data();
	createInfo.dependencyCount = dependencies.size();
	createInfo.pDependencies = dependencies.data();
	return m_device.createRenderPass(createInfo);
}

vk::Framebuffer Device::createFramebuffer(vk::RenderPass renderPass, vAP<vk::ImageView> attachments, vk::Extent2D extent, u32 layers) const {
	vk::FramebufferCreateInfo createInfo;
	createInfo.attachmentCount = attachments.size();
	createInfo.pAttachments = attachments.data();
	createInfo.renderPass = renderPass;
	createInfo.width = extent.width;
	createInfo.height = extent.height;
	createInfo.layers = layers;
	return m_device.createFramebuffer(createInfo);
}

void Device::defer(Deferred::Callback callback, u64 defer) {
	m_deferred.defer({std::move(callback), defer});
}

void Device::decrementDeferred() {
	m_deferred.decrement();
}
} // namespace le::graphics
