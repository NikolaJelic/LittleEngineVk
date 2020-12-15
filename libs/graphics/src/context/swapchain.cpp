#include <map>
#include <core/maths.hpp>
#include <graphics/context/device.hpp>
#include <graphics/context/swapchain.hpp>
#include <graphics/context/vram.hpp>

namespace le::graphics {
namespace {
template <typename T, typename U, typename V>
constexpr T bestFit(U&& all, V&& desired, T fallback) noexcept {
	for (auto const& d : desired) {
		if (std::find(all.begin(), all.end(), d) != all.end()) {
			return d;
		}
	}
	return fallback;
}

[[maybe_unused]] constexpr vk::Extent2D oriented(vk::Extent2D extent, vk::SurfaceTransformFlagBitsKHR transform) noexcept {
	if (transform & vk::SurfaceTransformFlagBitsKHR::eRotate90 || transform & vk::SurfaceTransformFlagBitsKHR::eRotate270) {
		return {extent.height, extent.width};
	}
	return extent;
}

struct SwapchainCreateInfo {
	SwapchainCreateInfo(vk::PhysicalDevice pd, vk::SurfaceKHR surface, Swapchain::CreateInfo const& info) : pd(pd), surface(surface) {
		vk::SurfaceCapabilitiesKHR capabilities = pd.getSurfaceCapabilitiesKHR(surface);
		std::vector<vk::SurfaceFormatKHR> colourFormats = pd.getSurfaceFormatsKHR(surface);
		availableModes = pd.getSurfacePresentModesKHR(surface);
		std::map<u32, vk::SurfaceFormatKHR> ranked;
		for (auto const& available : colourFormats) {
			u32 spaceRank = 0;
			for (auto desired : info.desired.colourSpaces) {
				if (desired == available.colorSpace) {
					break;
				}
				++spaceRank;
			}
			u32 formatRank = 0;
			for (auto desired : info.desired.colourFormats) {
				if (desired == available.format) {
					break;
				}
				++formatRank;
			}
			ranked.emplace(spaceRank + formatRank, available);
		}
		colourFormat = ranked.begin()->second;
		for (auto format : info.desired.depthFormats) {
			vk::FormatProperties const props = pd.getFormatProperties(format);
			static constexpr auto features = vk::FormatFeatureFlagBits::eDepthStencilAttachment;
			if ((props.optimalTilingFeatures & features) == features) {
				depthFormat = format;
				break;
			}
		}
		if (default_v(depthFormat)) {
			depthFormat = vk::Format::eD16Unorm;
		}
		presentMode = bestFit(availableModes, info.desired.presentModes, availableModes.front());
		imageCount = capabilities.minImageCount + 1;
		if (capabilities.maxImageCount > 0 && capabilities.maxImageCount < imageCount) {
			imageCount = capabilities.maxImageCount;
		}
		if (capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque) {
			compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
		} else if (capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eInherit) {
			compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eInherit;
		} else if (capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::ePreMultiplied) {
			compositeAlpha = vk::CompositeAlphaFlagBitsKHR::ePreMultiplied;
		} else {
			compositeAlpha = vk::CompositeAlphaFlagBitsKHR::ePostMultiplied;
		}
	}

	vk::Extent2D extent(glm::ivec2 fbSize) {
		vk::SurfaceCapabilitiesKHR capabilities = pd.getSurfaceCapabilitiesKHR(surface);
		current.transform = capabilities.currentTransform;
		current.extent = capabilities.currentExtent;
		if (!Swapchain::valid(fbSize) || current.extent.width != maths::max<u32>()) {
			return capabilities.currentExtent;
		} else {
			return vk::Extent2D(std::clamp((u32)fbSize.x, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
								std::clamp((u32)fbSize.y, capabilities.minImageExtent.height, capabilities.maxImageExtent.height));
		}
	}

	vk::PhysicalDevice pd;
	vk::SurfaceKHR surface;
	std::vector<vk::PresentModeKHR> availableModes;
	vk::SurfaceFormatKHR colourFormat;
	vk::Format depthFormat = {};
	vk::PresentModeKHR presentMode = {};
	vk::CompositeAlphaFlagBitsKHR compositeAlpha;
	Swapchain::Display current;
	u32 imageCount = 0;
};

void setFlags(Swapchain::Flags& out_flags, vk::Result result) {
	switch (result) {
	case vk::Result::eSuboptimalKHR: {
		if (!out_flags.test(Swapchain::Flag::eSuboptimal)) {
			g_log.log(lvl::debug, 0, "[{}] Vulkan swapchain is suboptimal", g_name);
		}
		out_flags.set(Swapchain::Flag::eSuboptimal);
		break;
	}
	case vk::Result::eErrorOutOfDateKHR: {
		if (!out_flags.test(Swapchain::Flag::eOutOfDate)) {
			g_log.log(lvl::debug, 0, "[{}] Vulkan swapchain is out of date", g_name);
		}
		out_flags.set(Swapchain::Flag::eOutOfDate);
		break;
	}
	default:
		break;
	}
}
} // namespace

Swapchain::Frame& Swapchain::Storage::frame() {
	return frames[imageIndex];
}

Swapchain::Swapchain(VRAM& vram) : m_vram(vram), m_device(vram.m_device) {
	if (!m_device.get().valid(m_device.get().m_metadata.surface)) {
		throw std::runtime_error("Invalid surface");
	}
	m_metadata.surface = m_device.get().m_metadata.surface;
}

Swapchain::Swapchain(VRAM& vram, CreateInfo const& info, glm::ivec2 framebufferSize) : Swapchain(vram) {
	m_metadata.info = info;
	if (!construct(framebufferSize)) {
		throw std::runtime_error("Failed to construct Vulkan swapchain");
	}
	makeRenderPass();
	auto const extent = m_storage.current.extent;
	auto const mode = presentModeName(m_metadata.presentMode);
	g_log.log(lvl::info, 1, "[{}] Vulkan swapchain constructed [{}x{}] [{}]", g_name, extent.width, extent.height, mode);
}

Swapchain::~Swapchain() {
	if (!default_v(m_storage.swapchain)) {
		g_log.log(lvl::info, 1, "[{}] Vulkan swapchain destroyed", g_name);
	}
	destroy(m_storage, true);
}

std::optional<RenderTarget> Swapchain::acquireNextImage(vk::Semaphore setDrawReady) {
	orientCheck();
	if (m_storage.flags.any(Flag::ePaused | Flag::eOutOfDate)) {
		return std::nullopt;
	}
	std::optional<vk::ResultValue<u32>> acquire;
	try {
		acquire = m_device.get().m_device.acquireNextImageKHR(m_storage.swapchain, maths::max<u64>(), setDrawReady, {});
		setFlags(m_storage.flags, acquire->result);
	} catch (vk::OutOfDateKHRError const& e) {
		m_storage.flags.set(Flag::eOutOfDate);
		g_log.log(lvl::warning, 1, "[{}] Swapchain failed to acquire next image [{}]", g_name, e.what());
		return std::nullopt;
	}
	if (!acquire || (acquire->result != vk::Result::eSuccess && acquire->result != vk::Result::eSuboptimalKHR)) {
		g_log.log(lvl::warning, 1, "[{}] Swapchain failed to acquire next image [{}]", g_name, acquire ? g_vkResultStr[acquire->result] : "Unknown Error");
		return std::nullopt;
	}
	m_storage.imageIndex = (u32)acquire->value;
	auto& frame = m_storage.frame();
	m_device.get().waitFor(frame.drawn);
	return frame.target;
}

bool Swapchain::present(vk::Semaphore drawWait, vk::Fence onDrawn) {
	if (m_storage.flags.any(Flag::ePaused | Flag::eOutOfDate)) {
		return false;
	}
	Frame& frame = m_storage.frame();
	vk::PresentInfoKHR presentInfo;
	auto const index = m_storage.imageIndex;
	presentInfo.waitSemaphoreCount = 1U;
	presentInfo.pWaitSemaphores = &drawWait;
	presentInfo.swapchainCount = 1U;
	presentInfo.pSwapchains = &m_storage.swapchain;
	presentInfo.pImageIndices = &index;
	vk::Result result;
	try {
		result = m_device.get().m_queues.present(presentInfo, false);
	} catch (vk::OutOfDateKHRError const& e) {
		g_log.log(lvl::warning, 1, "[{}] Swapchain Failed to present image [{}]", g_name, e.what());
		m_storage.flags.set(Flag::eOutOfDate);
		return false;
	}
	setFlags(m_storage.flags, result);
	if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
		g_log.log(lvl::warning, 1, "[{}] Swapchain Failed to present image [{}]", g_name, g_vkResultStr[result]);
		return false;
	}
	frame.drawn = onDrawn;
	orientCheck(); // Must submit acquired image, so skipping extent check here
	return true;
}

bool Swapchain::reconstruct(glm::ivec2 framebufferSize, Span<vk::PresentModeKHR> desiredModes) {
	if (!desiredModes.empty()) {
		m_metadata.info.desired.presentModes = desiredModes;
	}
	Storage retired = m_storage;
	m_metadata.retired = retired.swapchain;
	bool const bResult = construct(framebufferSize);
	auto const extent = m_storage.current.extent;
	auto const mode = presentModeName(m_metadata.presentMode);
	if (bResult) {
		g_log.log(lvl::info, 1, "[{}] Vulkan swapchain reconstructed [{}x{}] [{}]", g_name, extent.width, extent.height, mode);
	} else if (!m_storage.flags.test(Flag::ePaused)) {
		g_log.log(lvl::error, 1, "[{}] Vulkan swapchain reconstruction failed!", g_name);
	}
	destroy(retired, false);

	return bResult;
}

Swapchain::Flags Swapchain::flags() const noexcept {
	return m_storage.flags;
}

bool Swapchain::suboptimal() const noexcept {
	return m_storage.flags.test(Flag::eSuboptimal);
}

bool Swapchain::paused() const noexcept {
	return m_storage.flags.test(Flag::ePaused);
}

vk::RenderPass Swapchain::renderPass() const noexcept {
	return m_metadata.renderPass;
}

bool Swapchain::construct(glm::ivec2 framebufferSize) {
	m_storage = {};
	SwapchainCreateInfo info(m_device.get().m_physicalDevice, m_metadata.surface, m_metadata.info);
	m_metadata.availableModes = std::move(info.availableModes);
	{
		vk::SwapchainCreateInfoKHR createInfo;
		createInfo.minImageCount = info.imageCount;
		createInfo.imageFormat = info.colourFormat.format;
		createInfo.imageColorSpace = info.colourFormat.colorSpace;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
		auto const indices = m_device.get().m_queues.familyIndices(QType::eGraphics | QType::ePresent);
		createInfo.imageSharingMode = indices.size() == 1 ? vk::SharingMode::eExclusive : vk::SharingMode::eConcurrent;
		createInfo.pQueueFamilyIndices = indices.data();
		createInfo.queueFamilyIndexCount = (u32)indices.size();
		createInfo.compositeAlpha = info.compositeAlpha;
		m_metadata.presentMode = createInfo.presentMode = info.presentMode;
		createInfo.clipped = vk::Bool32(true);
		createInfo.surface = m_metadata.surface;
		createInfo.oldSwapchain = m_metadata.retired;
		createInfo.imageExtent = info.extent(framebufferSize);
		createInfo.preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
		if (createInfo.imageExtent.width <= 0 || createInfo.imageExtent.height <= 0) {
			m_storage.flags.set(Flag::ePaused);
			return false;
		}
		m_storage.current = info.current;
		m_storage.swapchain = m_device.get().m_device.createSwapchainKHR(createInfo);
		m_metadata.formats.colour = info.colourFormat.format;
		m_metadata.formats.depth = info.depthFormat;
		if (!m_metadata.original) {
			m_metadata.original = info.current;
		}
		m_metadata.retired = vk::SwapchainKHR();
	}
	{
		auto images = m_device.get().m_device.getSwapchainImagesKHR(m_storage.swapchain);
		m_storage.frames.reserve(images.size());
		Image::CreateInfo depthImageInfo;
		depthImageInfo.createInfo.format = info.depthFormat;
		depthImageInfo.vmaUsage = VMA_MEMORY_USAGE_GPU_ONLY;
		depthImageInfo.createInfo.extent = vk::Extent3D(m_storage.current.extent, 1);
		depthImageInfo.createInfo.tiling = vk::ImageTiling::eOptimal;
		depthImageInfo.createInfo.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
		depthImageInfo.createInfo.samples = vk::SampleCountFlagBits::e1;
		depthImageInfo.createInfo.imageType = vk::ImageType::e2D;
		depthImageInfo.createInfo.initialLayout = vk::ImageLayout::eUndefined;
		depthImageInfo.createInfo.mipLevels = 1;
		depthImageInfo.createInfo.arrayLayers = 1;
		depthImageInfo.queueFlags = QType::eGraphics;
		depthImageInfo.name = "swapchain_depth";
		m_storage.depthImage = m_vram.get().construct(depthImageInfo);
		m_storage.depthImageView = m_device.get().createImageView(m_storage.depthImage.image, info.depthFormat, vk::ImageAspectFlagBits::eDepth);
		auto const format = info.colourFormat.format;
		auto const aspectFlags = vk::ImageAspectFlagBits::eColor;
		for (auto const& image : images) {
			Frame frame;
			frame.target.colour.image = image;
			frame.target.depth.image = m_storage.depthImage.image;
			frame.target.colour.view = m_device.get().createImageView(image, format, aspectFlags);
			frame.target.depth.view = m_storage.depthImageView;
			frame.target.extent = m_storage.current.extent;
			ENSURE(frame.target.extent.width > 0 && frame.target.extent.height > 0, "Invariant violated");
			m_storage.frames.push_back(std::move(frame));
		}
		if (m_storage.frames.empty()) {
			throw std::runtime_error("Failed to construct Vulkan swapchain!");
		}
	}
	return true;
}

void Swapchain::makeRenderPass() {
	std::array<vk::AttachmentDescription, 2> attachments;
	vk::AttachmentReference colourAttachment, depthAttachment;
	{
		attachments[0].format = m_metadata.formats.colour;
		attachments[0].samples = vk::SampleCountFlagBits::e1;
		attachments[0].loadOp = vk::AttachmentLoadOp::eClear;
		attachments[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attachments[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attachments[0].initialLayout = vk::ImageLayout::eUndefined;
		attachments[0].finalLayout = vk::ImageLayout::ePresentSrcKHR;
		colourAttachment.attachment = 0;
		colourAttachment.layout = vk::ImageLayout::eColorAttachmentOptimal;
	}
	{
		attachments[1].format = m_metadata.formats.depth;
		attachments[1].samples = vk::SampleCountFlagBits::e1;
		attachments[1].loadOp = vk::AttachmentLoadOp::eClear;
		attachments[1].storeOp = vk::AttachmentStoreOp::eDontCare;
		attachments[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attachments[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attachments[1].initialLayout = vk::ImageLayout::eUndefined;
		attachments[1].finalLayout = vk::ImageLayout::ePresentSrcKHR;
		depthAttachment.attachment = 1;
		depthAttachment.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
	}
	vk::SubpassDescription subpass;
	subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colourAttachment;
	subpass.pDepthStencilAttachment = &depthAttachment;
	vk::SubpassDependency dependency;
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
	m_metadata.renderPass = m_device.get().createRenderPass(attachments, subpass, dependency);
}

void Swapchain::destroy(Storage& out_storage, bool bMeta) {
	auto r = bMeta ? std::exchange(m_metadata.renderPass, vk::RenderPass()) : vk::RenderPass();
	m_device.get().waitIdle();
	auto lock = m_device.get().m_queues.lock();
	for (auto& frame : out_storage.frames) {
		m_device.get().destroy(frame.target.colour.view);
	}
	m_device.get().destroy(out_storage.depthImageView, out_storage.swapchain, r);
	m_vram.get().destroy(out_storage.depthImage);
	out_storage = {};
}

void Swapchain::orientCheck() {
	auto const capabilities = m_device.get().m_physicalDevice.getSurfaceCapabilitiesKHR(m_metadata.surface);
	if (capabilities.currentTransform != m_storage.current.transform) {
		using vkst = vk::SurfaceTransformFlagBitsKHR;
		auto const c = capabilities.currentTransform;
		if (m_metadata.original->transform == vkst::eIdentity || m_metadata.original->transform == vkst::eRotate180) {
			m_storage.flags[Flag::eRotated] = c == vkst::eRotate90 || c == vkst::eRotate270;
		} else if (m_metadata.original->transform == vkst::eRotate90 || m_metadata.original->transform == vkst::eRotate270) {
			m_storage.flags[Flag::eRotated] = c == vkst::eIdentity || c == vkst::eRotate180;
		}
		m_storage.current.transform = capabilities.currentTransform;
	}
	if (capabilities.currentExtent != maths::max<u32>() && capabilities.currentExtent != m_storage.current.extent) {
		m_storage.flags.set(Flag::eOutOfDate);
	}
}
} // namespace le::graphics
