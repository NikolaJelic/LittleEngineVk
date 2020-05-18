#pragma once
#include <deque>
#include <vector>
#include <glm/glm.hpp>
#include "core/delegate.hpp"
#include "engine/window/common.hpp"
#include "engine/gfx/light.hpp"
#include "engine/gfx/pipeline.hpp"
#include "engine/gfx/renderer.hpp"
#include "presenter.hpp"
#include "pipeline_impl.hpp"
#include "resource_descriptors.hpp"

namespace le
{
class Transform;
class WindowImpl;
} // namespace le

namespace le::gfx
{
class Mesh;
class Renderer;

class RendererImpl final
{
public:
	struct Info final
	{
		PresenterInfo presenterInfo;
		WindowID windowID;
		u8 frameCount = 2;
	};

private:
	struct FrameSync final
	{
		rd::Set set;
		vk::Semaphore renderReady;
		vk::Semaphore presentReady;
		vk::Fence drawing;
		vk::Framebuffer framebuffer;
		vk::CommandBuffer commandBuffer;
		vk::CommandPool commandPool;
	};

public:
	std::string m_name;

private:
	Presenter m_presenter;
	vk::DescriptorPool m_descriptorPool;
	std::deque<Pipeline> m_pipelines;
	std::vector<FrameSync> m_frames;
	Renderer* m_pRenderer;
	struct
	{
		Pipeline* pDefault = nullptr;
		Pipeline* pSkybox = nullptr;
	} m_pipes;

	u64 m_drawnFrames = 0;
	u32 m_maxDiffuseID = 0;
	u32 m_maxSpecularID = 0;
	size_t m_index = 0;
	WindowID m_window;
	u8 m_frameCount = 0;

public:
	RendererImpl(Info const& info, Renderer* pOwner);
	~RendererImpl();

public:
	void create(u8 frameCount = 2);
	void destroy();

	Pipeline* createPipeline(Pipeline::Info info);

	void update();
	bool render(Renderer::Scene scene);

public:
	vk::Viewport transformViewport(ScreenRect const& nRect = {}, glm::vec2 const& depth = {0.0f, 1.0f}) const;
	vk::Rect2D transformScissor(ScreenRect const& nRect = {}) const;

	u64 framesDrawn() const;
	u8 virtualFrameCount() const;

private:
	void onFramebufferResize();
	FrameSync& frameSync();
	void next();

	friend class le::WindowImpl;
};
} // namespace le::gfx