#pragma once
#include <optional>
#include <core/ref.hpp>
#include <core/span.hpp>
#include <glm/vec2.hpp>
#include <graphics/types.hpp>

namespace le::graphics {
class VRAM;
class Device;

class Swapchain {
  public:
	enum class Flag : s8 { ePaused, eOutOfDate, eSuboptimal, eRotated, eCOUNT_ };
	using Flags = kt::enum_flags<Flag>;

	struct Frame {
		RenderTarget target;
		vk::Fence drawn;
	};
	struct Display {
		vk::Extent2D extent = {};
		vk::SurfaceTransformFlagBitsKHR transform = {};
	};
	struct Storage {
		Image depthImage;
		vk::ImageView depthImageView;
		vk::SwapchainKHR swapchain;
		std::vector<Frame> frames;

		Display current;
		u32 imageIndex = 0;
		u8 imageCount = 0;
		Flags flags;

		Frame& frame();
	};
	struct CreateInfo {
		static constexpr auto defaultColourSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
		static constexpr auto defaultColourFormat = vk::Format::eB8G8R8A8Srgb;
		static constexpr std::array defaultDepthFormats = {vk::Format::eD32SfloatS8Uint, vk::Format::eD32Sfloat, vk::Format::eD24UnormS8Uint};
		static constexpr auto defaultPresentMode = vk::PresentModeKHR::eFifo;

		struct {
			Span<vk::ColorSpaceKHR> colourSpaces = defaultColourSpace;
			Span<vk::Format> colourFormats = defaultColourFormat;
			Span<vk::Format> depthFormats = defaultDepthFormats;
			Span<vk::PresentModeKHR> presentModes = defaultPresentMode;
			u32 imageCount = 2;
		} desired;
	};
	struct Metadata {
		CreateInfo info;
		vk::RenderPass renderPass;
		vk::SurfaceKHR surface;
		vk::SwapchainKHR retired;
		vk::PresentModeKHR presentMode;
		std::optional<Display> original;
		std::vector<vk::PresentModeKHR> availableModes;
		struct {
			vk::Format colour;
			vk::Format depth;
		} formats;
	};

	static constexpr std::string_view presentModeName(vk::PresentModeKHR mode) noexcept;
	static constexpr bool valid(glm::ivec2 framebufferSize) noexcept;

	Swapchain(VRAM& vram);
	Swapchain(VRAM& vram, CreateInfo const& info, glm::ivec2 framebufferSize = {});
	Swapchain(Swapchain&&);
	Swapchain& operator=(Swapchain&&);
	~Swapchain();

	std::optional<RenderTarget> acquireNextImage(vk::Semaphore setDrawReady);
	bool present(vk::Semaphore drawWait, vk::Fence onDrawn);
	bool reconstruct(glm::ivec2 framebufferSize = {}, Span<vk::PresentModeKHR> desiredModes = {});

	Flags flags() const noexcept;
	bool suboptimal() const noexcept;
	bool paused() const noexcept;
	vk::RenderPass renderPass() const noexcept;

	Storage m_storage;
	Metadata m_metadata;
	Ref<VRAM> m_vram;
	Ref<Device> m_device;

  private:
	bool construct(glm::ivec2 framebufferSize);
	void makeRenderPass();
	void destroy(Storage& out_storage, bool bMeta);
	void orientCheck();
};

// impl

inline constexpr std::string_view Swapchain::presentModeName(vk::PresentModeKHR mode) noexcept {
	switch (mode) {
	case vk::PresentModeKHR::eFifo:
		return "FIFO";
	case vk::PresentModeKHR::eFifoRelaxed:
		return "FIFO Relaxed";
	case vk::PresentModeKHR::eImmediate:
		return "Immediate";
	case vk::PresentModeKHR::eMailbox:
		return "Mailbox";
	default:
		return "Other";
	}
}

inline constexpr bool Swapchain::valid(glm::ivec2 framebufferSize) noexcept {
	return framebufferSize.x > 0 && framebufferSize.y > 0;
}
} // namespace le::graphics
