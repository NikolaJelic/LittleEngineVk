#pragma once
#include <core/traits.hpp>
#include <core/utils.hpp>
#include <graphics/context/defer_queue.hpp>
#include <graphics/context/memory.hpp>
#include <graphics/context/transfer.hpp>

namespace le::graphics {
class Device;
using LayoutTransition = std::pair<vk::ImageLayout, vk::ImageLayout>;

class VRAM final : public Memory {
  public:
	using notify_t = Transfer::notify_t;
	using Future = ::le::utils::Future<notify_t>;

	VRAM(Device& device, Transfer::CreateInfo const& transferInfo = {});
	~VRAM();

	View<Buffer> createBO(std::string_view name, vk::DeviceSize size, vk::BufferUsageFlags usage, bool bHostVisible);

	[[nodiscard]] Future copy(CView<Buffer> src, View<Buffer> dst, vk::DeviceSize size = 0);
	[[nodiscard]] Future stage(View<Buffer> deviceBuffer, void const* pData, vk::DeviceSize size = 0);
	[[nodiscard]] Future copy(Span<Span<std::byte>> pixelsArr, View<Image> dst, LayoutTransition layouts);

	void defer(View<Buffer> buffer, u64 defer = Deferred::defaultDefer);
	void defer(View<Image> image, u64 defer = Deferred::defaultDefer);

	template <typename Cont = std::initializer_list<Ref<Future const>>>
	void wait(Cont&& futures) const;
	void waitIdle();

	Transfer m_transfer;
	Ref<Device> m_device;
	struct {
		vk::PipelineStageFlags stages;
		vk::AccessFlags access;
	} m_post;
};

// impl

template <typename Cont>
void VRAM::wait(Cont&& futures) const {
	for (Future const& f : futures) {
		f.wait();
	}
}
} // namespace le::graphics
