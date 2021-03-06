#pragma once
#include <deque>
#include <memory>
#include <vector>
#include <core/transform.hpp>
#include <engine/gfx/camera.hpp>
#include <engine/gfx/light.hpp>
#include <engine/gfx/pipeline.hpp>
#include <engine/gfx/viewport.hpp>
#include <engine/resources/resource_types.hpp>

namespace le {
class Transform;
class WindowImpl;
} // namespace le

namespace le::gfx::render {
class Driver final {
  public:
	struct ClearValues final {
		glm::vec2 depthStencil = {1.0f, 0.0f};
		Colour colour = colours::black;
	};

	struct Skybox final {
		res::Texture cubemap;
		Pipeline pipeline;
	};

	struct Drawable final {
		std::vector<res::Mesh> meshes;
		Ref<Transform const> transform = Transform::s_identity;
		Pipeline pipeline;
	};

	struct Batch final {
		ScreenRect viewport;
		ScreenRect scissor;
		std::deque<Drawable> drawables;
		bool bIgnoreGameView = false;
	};

	struct View final {
		glm::mat4 mat_vp = {};
		glm::mat4 mat_v = {};
		glm::mat4 mat_p = {};
		glm::mat4 mat_ui = {};
		glm::vec3 pos_v = {};
		Skybox skybox;
	};

	struct Scene final {
		View view;
		ClearValues clear;
		std::deque<Batch> batches;
		std::vector<DirLight> dirLights;
	};

	struct Stats final {
		u64 trisDrawn = 0;
	};

  public:
	static std::string const s_tName;

  public:
	Stats m_stats;

  public:
	Driver();
	Driver(Driver&&);
	Driver& operator=(Driver&&);
	~Driver();

  public:
	void submit(Scene scene, ScreenRect const& sceneView);

	glm::vec2 screenToN(glm::vec2 const& screenXY) const;
	ScreenRect clampToView(glm::vec2 const& screenXY, glm::vec2 const& nRect, glm::vec2 const& padding = {}) const;
	void fill(View& out_view, Viewport const& viewport, Camera const& camera, f32 orthoDepth = 2.0f) const;

  private:
	void render(bool bEditor);

  private:
	class Impl;
	friend class le::WindowImpl;
	friend class Impl;

  private:
	std::unique_ptr<class Impl> m_uImpl;
	Scene m_scene;
	ScreenRect m_sceneView;
};
} // namespace le::gfx::render
