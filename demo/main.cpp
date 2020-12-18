#include <engine/levk.hpp>
#include <level.hpp>

#include <core/index_view.hpp>
#include <graphics/bitmap_text.hpp>
#include <graphics/context/bootstrap.hpp>
#include <graphics/geometry.hpp>
#include <graphics/mesh.hpp>
#include <graphics/render_context.hpp>
#include <graphics/shader.hpp>
#include <graphics/texture.hpp>
#include <graphics/utils/utils.hpp>
#include <window/desktop_instance.hpp>

using namespace le;

enum class Flag { eRecreated, eResized, ePaused, eClosed, eDebug0, eCOUNT_ };
using Flags = kt::enum_flags<Flag>;

static void poll(Flags& out_flags, window::EventQueue queue) {
	while (auto e = queue.pop()) {
		switch (e->type) {
		case window::Event::Type::eClose: {
			out_flags.set(Flag::eClosed);
			break;
		}
		case window::Event::Type::eInput: {
			auto const& input = e->payload.input;
			if (input.key == window::Key::eW && input.action == window::Action::eRelease && input.mods[window::Mod::eControl]) {
				out_flags.set(Flag::eClosed);
			}
			if (input.key == window::Key::eD && input.action == window::Action::eRelease && input.mods[window::Mod::eControl]) {
				out_flags.flip(Flag::eDebug0);
			}
			break;
		}
		case window::Event::Type::eSuspend: {
			out_flags[Flag::ePaused] = e->payload.bSet;
			break;
		}
		case window::Event::Type::eResize: {
			auto const& resize = e->payload.resize;
			if (resize.bFramebuffer) {
				out_flags.set(Flag::eResized);
			}
			break;
		}
		default:
			break;
		}
	}
}

struct GPULister : os::ICmdArg {
	inline static constexpr std::array names = {"gpu-list"sv, "list-gpus"sv};

	Span<std::string_view> keyVariants() const override {
		return names;
	}

	bool halt(std::string_view) override {
		graphics::g_log.minVerbosity = LibLogger::Verbosity::eEndUser;
		graphics::Instance inst(graphics::Instance::CreateInfo{});
		std::stringstream str;
		str << "Available GPUs:\n";
		int i = 0;
		for (auto const& d : inst.availableDevices(graphics::Device::requiredExtensions)) {
			str << '\t' << i++ << ". " << d << "\n";
		}
		str << "\n";
		std::cout << str.str();
		return true;
	}

	Usage usage() const override {
		return {"", "List supported GPUs"};
	}
};

struct GPUPicker : os::ICmdArg {
	inline static constexpr std::array names = {"use-gpu"sv, "pick-gpu"sv};

	inline static std::optional<std::size_t> s_picked;

	Span<std::string_view> keyVariants() const override {
		return names;
	}

	bool halt(std::string_view params) override {
		s32 idx = utils::strings::toS32(params, -1);
		if (idx >= 0) {
			s_picked = (std::size_t)idx;
			logD("Using custom GPU index: {}", idx);
		}
		return false;
	}

	Usage usage() const override {
		return {"<0-...>", "Select a custom available GPU"};
	}
};

void listCmdArgs();

struct HelpCmd : os::ICmdArg {
	inline static constexpr std::array names = {"h"sv, "help"sv};

	Span<std::string_view> keyVariants() const override {
		return names;
	}

	bool halt(std::string_view) override {
		listCmdArgs();
		return true;
	}

	Usage usage() const override {
		return {"", "List all command line arguments"};
	}
};

struct Sets {
	std::unordered_map<u32, graphics::SetFactory> sets;

	void make(Span<u32> setNumbers, graphics::Pipeline const& pipe) {
		for (u32 num : setNumbers) {
			sets.emplace(num, pipe.makeSetFactory(num));
		}
	}

	graphics::SetFactory& operator[](u32 set) {
		if (auto it = sets.find(set); it != sets.end()) {
			return it->second;
		}
		ENSURE(false, "Nonexistent set");
		throw std::runtime_error("Nonexistent set");
	}

	void swap() {
		for (auto& [_, set] : sets) {
			set.swap();
		}
	}
};

struct SetLayouts {
	std::unordered_map<Hash, Sets> sets;

	void make(Hash layout, Span<u32> setNumbers, graphics::Pipeline const& pipe) {
		sets[layout].make(setNumbers, pipe);
	}

	Sets& operator[](Hash hash) {
		if (auto it = sets.find(hash); it != sets.end()) {
			return it->second;
		}
		ENSURE(false, "Nonexistent layout");
		throw std::runtime_error("Nonexistent layout");
	}

	void swap() {
		for (auto& [_, s] : sets) {
			s.swap();
		}
	}
};

struct Material {
	virtual void write(graphics::DescriptorSet&) {
	}
	virtual void bind(graphics::CommandBuffer&, graphics::Pipeline const&, graphics::DescriptorSet const&) const {
	}
};

struct TexturedMaterial : Material {
	CView<graphics::Texture> diffuse;
	u32 binding = 0;

	void write(graphics::DescriptorSet& ds) override {
		ENSURE(diffuse, "Null pipeline/texture view");
		ds.updateTextures(binding, *diffuse);
	}
	void bind(graphics::CommandBuffer& cb, graphics::Pipeline const& pi, graphics::DescriptorSet const& ds) const override {
		ENSURE(diffuse, "Null texture view");
		cb.bindSets(pi.layout(), ds.get(), ds.setNumber());
	}
};

struct Prop2 {
	Ref<Transform> transform;
	Ref<graphics::Mesh> mesh;
	Ref<Material> material;
};

struct Font {
	io::Path atlasID;
	io::Path samplerID;
	io::Path materialID;
	std::optional<graphics::Texture> atlas;
	std::array<graphics::Glyph, maths::max<u8>()> glyphs;

	graphics::Glyph deserialise(u8 c, dj::object const& json) {
		graphics::Glyph ret;
		ret.ch = c;
		ret.st = {(s32)json.value<dj::integer>("x"), (s32)json.value<dj::integer>("y")};
		ret.uv = ret.cell = {(s32)json.value<dj::integer>("width"), (s32)json.value<dj::integer>("height")};
		ret.offset = {(s32)json.value<dj::integer>("originX"), (s32)json.value<dj::integer>("originY")};
		auto const pAdvance = json.find<dj::integer>("advance");
		ret.xAdv = pAdvance ? (s32)pAdvance->value : ret.cell.x;
		ret.orgSizePt = (s32)json.value<dj::integer>("size");
		ret.bBlank = json.value<dj::boolean>("isBlank");
		return ret;
	}

	void deserialise(dj::object const& json) {
		if (auto pAtlas = json.find<dj::string>("sheetID")) {
			atlasID = pAtlas->value;
		}
		if (auto pSampler = json.find<dj::string>("samplerID")) {
			samplerID = pSampler->value;
		}
		if (auto pMaterial = json.find<dj::string>("materialID")) {
			materialID = pMaterial->value;
		}
		if (auto pGlyphsData = json.find<dj::object>("glyphs")) {
			for (auto& [key, value] : pGlyphsData->fields) {
				if (!key.empty() && value->type() == dj::data_type::object) {
					graphics::Glyph const glyph = deserialise((u8)key[0], *value->cast<dj::object>());
					if (glyph.cell.x > 0 && glyph.cell.y > 0) {
						glyphs[(std::size_t)glyph.ch] = glyph;
					} else {
						logW("Could not deserialise Glyph '{}'!", key[0]);
					}
				}
			}
		}
	}

	bool create(graphics::VRAM& vram, io::Reader const& reader, io::Path const& id, io::Path const& path, vk::Sampler sampler, vk::Format format) {
		auto jsonText = reader.string(path);
		if (!jsonText) {
			return false;
		}
		dj::object json;
		if (!json.read(*jsonText)) {
			return false;
		}
		deserialise(json);
		auto bytes = reader.bytes(path.parent_path() / atlasID);
		if (!bytes) {
			return false;
		}
		atlas = graphics::Texture((id / "atlas").generic_string(), vram);
		graphics::Texture::CreateInfo info;
		info.sampler = sampler;
		info.data = graphics::Texture::Compressed{{*bytes}};
		info.format = format;
		if (!atlas->construct(info)) {
			return false;
		}
		return true;
	}
};

struct Text {
	graphics::BitmapText text;
	std::optional<graphics::Mesh> mesh;
	glm::mat4 model = glm::mat4(1.0f);

	void create(graphics::VRAM& vram, io::Path const& id) {
		mesh = graphics::Mesh((id / "mesh").generic_string(), vram);
	}

	bool set(Font const& font, std::string_view str) {
		text.text = str;
		if (mesh) {
			return mesh->construct(text.generate(font.glyphs, font.atlas->data().size));
		}
		return false;
	}
};

struct VP {
	glm::mat4 mat_p;
	glm::mat4 mat_v;
	glm::mat4 mat_ui;
};

struct Skybox {
	CView<graphics::Mesh> mesh;
	CView<graphics::Texture> cubemap;

	bool ready() const {
		return cubemap && cubemap->ready();
	}
	void update(graphics::DescriptorSet& set, CView<graphics::Buffer> vp) const {
		if (ready()) {
			set.updateBuffers(0, vp, sizeof(VP));
			set.updateTextures(1, *cubemap);
		}
	}
	void draw(graphics::CommandBuffer& cb, graphics::Pipeline const& pi, graphics::DescriptorSet const& set) {
		if (ready()) {
			cb.bindPipe(pi);
			cb.bindSets(pi.layout(), set.get(), set.setNumber());
			cb.bindVBO(mesh->vbo().buffer, mesh->ibo().buffer);
			cb.drawIndexed(mesh->ibo().count);
		}
	}
};
struct Scene {
	Skybox skybox;
	std::unordered_map<Ref<graphics::Pipeline>, std::vector<Prop2>> props;
};

GPULister g_gpuLister;
GPUPicker g_gpuPicker;
HelpCmd g_help;
std::array<Ref<os::ICmdArg>, 3> const g_cmdArgs = {g_gpuLister, g_gpuPicker, g_help};

void listCmdArgs() {
	std::stringstream str;
	for (os::ICmdArg const& arg : g_cmdArgs) {
		str << '[';
		bool bFirst = true;
		for (auto key : arg.keyVariants()) {
			if (!bFirst) {
				str << ", ";
			}
			bFirst = false;
			str << (key.length() == 1 ? "-"sv : "--"sv) << key;
		}
		auto const u = arg.usage();
		if (!u.params.empty()) {
			str << '=' << u.params;
		}
		str << "] : " << u.summary << '\n';
	}
	std::cout << str.str();
}

int main(int argc, char** argv) {
	try {
		os::args({argc, argv});
		if (os::halt(g_cmdArgs)) {
			return 0;
		}
		io::FileReader reader;
		io::Path const prefix = os::dirPath(os::Dir::eWorking) / "data";
		reader.mount(prefix);
		reader.mount(os::dirPath(os::Dir::eWorking) / "demo/data");
		auto testV = graphics::utils::compileGlsl("shaders/test.vert", {}, prefix);
		auto uiV = graphics::utils::compileGlsl("shaders/ui.vert", {}, prefix);
		auto uiF = graphics::utils::compileGlsl("shaders/ui.frag", {}, prefix);
		auto testF = graphics::utils::compileGlsl("shaders/test.frag", {}, prefix);
		auto testFTex = graphics::utils::compileGlsl("shaders/test_tex.frag", {}, prefix);
		auto skyV = graphics::utils::compileGlsl("shaders/skybox.vert", {}, prefix);
		auto skyF = graphics::utils::compileGlsl("shaders/skybox.frag", {}, prefix);
		auto vert = reader.bytes("shaders/uber.vert.spv");
		auto frag = reader.bytes("shaders/uber.frag.spv");
		auto tex0 = reader.bytes("textures/container2.png");
		auto const cubemap = graphics::utils::loadCubemap(reader, "skyboxes/sky_dusk");
		window::CreateInfo winInfo;
		winInfo.config.title = "levk demo";
		winInfo.config.size = {1280, 720};
		winInfo.options.bCentreCursor = true;
		winInfo.options.verbosity = LibLogger::Verbosity::eLibrary;
		window::DesktopInstance winst(winInfo);
		auto makeSurface = [&winst](vk::Instance instance) -> vk::SurfaceKHR {
			vk::SurfaceKHR surface;
			winst.vkCreateSurface(instance, surface);
			return surface;
		};
		graphics::Bootstrap::CreateInfo bootInfo;
		bootInfo.instance.extensions = winst.vkInstanceExtensions();
		bootInfo.instance.bValidation = levk_debug;
		bootInfo.instance.validationLog = dl::level::info;
		bootInfo.logVerbosity = LibLogger::Verbosity::eLibrary;
		bootInfo.device.pickOverride = GPUPicker::s_picked;
		// bootInfo.device.qselect = graphics::Device::QSelect::eSingleFamily;
		graphics::Bootstrap boot(bootInfo, makeSurface, winst.framebufferSize());
		boot.vram.m_bLogAllocs = true;
		graphics::RenderContext context(boot.swapchain);
		{
			graphics::Shader shader(boot.device);
			if (vert && frag) {
				shader.reconstruct({std::move(*vert), std::move(*frag)});
				logD("uber shader created");
			}

			graphics::Geometry gcube = graphics::makeCube(0.5f);
			auto const skyCubeI = gcube.indices;
			auto const skyCubeV = gcube.positions();
			VP vp;
			graphics::Mesh mesh0("cube", boot.vram, graphics::Mesh::Type::eStatic);
			graphics::Mesh mesh1("cone", boot.vram, graphics::Mesh::Type::eStatic);
			graphics::Mesh skyCube("sky_cube", boot.vram, graphics::Mesh::Type::eStatic);
			vk::Sampler sampler = context.makeSampler(context.samplerInfo({vk::Filter::eLinear, vk::Filter::eLinear}));
			graphics::Texture texC("container", boot.vram), texR("red", boot.vram), sky("sky_dusk", boot.vram);
			graphics::Texture::CreateInfo texInfo;
			graphics::Texture::Raw raw;
			raw.bytes = graphics::utils::convert({0xff, 0, 0, 0xff});
			raw.size = {1, 1};
			graphics::Texture::Compressed comp = {{std::move(*tex0)}};
			graphics::Texture::Compressed cm = {std::move(cubemap)};
			texInfo.data = std::move(comp);
			texInfo.sampler = sampler;
			texC.construct(texInfo);
			texInfo.data = raw;
			texR.construct(texInfo);
			texInfo.data = std::move(cm);
			sky.construct(texInfo);
			mesh0.construct(gcube);
			mesh1.construct(graphics::makeCone());
			skyCube.construct(Span(skyCubeV), skyCubeI);
			if (!testV || !testF || !testFTex || !skyV || !skyF) {
				logE("shaders missing");
				return 1;
			}

			Font font;
			font.create(boot.vram, reader, "fonts/default", "fonts/default.json", sampler, context.colourFormat());

			auto test = graphics::Shader(boot.device, {*reader.bytes(*testV), *reader.bytes(*testF)});
			auto testTex = graphics::Shader(boot.device, {*reader.bytes(*testV), *reader.bytes(*testFTex)});
			auto sskybox = graphics::Shader(boot.device, {*reader.bytes(*skyV), *reader.bytes(*skyF)});
			graphics::Shader ui(boot.device, {*reader.bytes(*uiV), *reader.bytes(*uiF)});
			auto pipe = context.makePipeline("test", context.pipeInfo(test));
			auto pipeTex = context.makePipeline("test_tex", context.pipeInfo(testTex, graphics::PFlags::inverse()));
			auto pipeUI = context.makePipeline("ui", context.pipeInfo(ui, graphics::PFlags::inverse()));
			Text text;
			text.create(boot.vram, "text");
			text.text.size = 80U;
			text.text.colour = colours::yellow;
			text.text.pos = {0.0f, 200.0f, 0.0f};
			text.set(font, "Hi!");

			SetLayouts layouts;
			std::array const setNums = {0U, 1U, 2U};
			layouts.make("main", setNums, pipeTex);
			auto pipeSkyInfo = context.pipeInfo(sskybox);
			pipeSkyInfo.fixedState.depthStencilState.depthWriteEnable = false;
			pipeSkyInfo.fixedState.vertexInput = context.vertexInput({0, sizeof(glm::vec3), {{vk::Format::eR32G32B32Sfloat, 0}}});
			auto pipeSky = context.makePipeline("skybox", pipeSkyInfo);
			layouts.make("skybox", 0, pipeSky);
			texC.wait();
			texR.wait();
			winst.show();
			Flags flags;
			std::array<Transform, 4> tr;
			tr[1].position({-5.0f, -1.0f, -2.0f});
			tr[2].position({0.0f, -2.0f, -3.0f});
			time::Point t = time::now();
			glm::vec3 camPos = {0.0f, 2.0f, 4.0f};
			while (true) {
				Time_s dt = time::now() - t;
				t = time::now();
				poll(flags, winst.pollEvents());
				if (flags.test(Flag::eClosed)) {
					break;
				}
				if (flags.test(Flag::ePaused)) {
					continue;
				}
				if (flags.test(Flag::eResized)) {
					/*if (!context.recreated(winst.framebufferSize())) {
						ENSURE(false, "Swapchain failure");
					}*/
					flags.reset(Flag::eResized);
				}
				if (context.reconstructed(winst.framebufferSize())) {
					continue;
				}

				// tick
				threads::sleep(5ms);
				auto const fb = winst.framebufferSize();
				vp.mat_p = glm::perspective(glm::radians(45.0f), (f32)fb.x / std::max((f32)fb.y, 1.0f), 0.1f, 100.0f);
				{
					f32 const w = (f32)fb.x * 0.5f;
					f32 const h = (f32)fb.y * 0.5f;
					vp.mat_ui = glm::ortho(-w, w, -h, h, -1.0f, 1.0f);
				}
				// camera
				{
					glm::vec3 const moveDir = glm::normalize(glm::cross(camPos, graphics::g_nUp));
					camPos += moveDir * dt.count() * 0.75f;
					vp.mat_v = glm::lookAt(camPos, {}, graphics::g_nUp);
				}
				tr[0].rotate(glm::radians(-180.0f) * dt.count(), glm::normalize(glm::vec3(1.0f)));
				tr[1].rotate(glm::radians(360.0f) * dt.count(), graphics::g_nUp);

				TexturedMaterial texMat;
				texMat.diffuse = texC;
				TexturedMaterial fontMat;
				fontMat.diffuse = *font.atlas;
				Material mat;
				Scene scene;
				scene.skybox.mesh = skyCube;
				scene.skybox.cubemap = sky;
				scene.props[*pipeTex].push_back(Prop2{tr[0], mesh0, texMat});
				scene.props[*pipe].push_back(Prop2{tr[1], mesh0, mat});
				scene.props[*pipe].push_back(Prop2{tr[2], mesh1, mat});
				scene.props[*pipeUI].push_back(Prop2{tr[3], *text.mesh, fontMat});

				// render
				if (context.waitForFrame()) {
					// write / update
					auto& smain = layouts["main"];
					auto& ssky = layouts["skybox"];
					smain[0].front().writeBuffer(0, vp);
					scene.skybox.update(ssky[0].front(), smain[0].front().buffers(0).front());
					std::size_t idx = 0;
					for (auto& [p, props] : scene.props) {
						for (auto& prop : props) {
							Material& mat = prop.material;
							Transform& t = prop.transform;
							smain[1].at(idx).writeBuffer(0, t.model());
							mat.write(smain[2].at(idx));
							++idx;
						}
					}
					// draw
					if (auto r = context.render(Colour(0x040404ff))) {
						auto& cb = r->primary();
						cb.setViewportScissor(context.viewport(), context.scissor());
						scene.skybox.draw(cb, pipeSky, ssky[0].front());
						cb.bindSets(pipe->layout(), smain[0].front().get(), smain[0].front().setNumber());
						std::size_t idx = 0;
						for (auto& [p, props] : scene.props) {
							graphics::Pipeline& pi = p;
							cb.bindPipe(pi);
							for (auto const& prop : props) {
								Material& mat = prop.material;
								graphics::Mesh& m = prop.mesh;
								cb.bindSets(pipe->layout(), smain[1].at(idx).get(), smain[1].at(idx).setNumber());
								mat.bind(cb, pi, smain[2].at(idx));
								cb.bindVBO(m.vbo().buffer, m.ibo().buffer);
								if (m.hasIndices()) {
									cb.drawIndexed(m.ibo().count);
								} else {
									cb.draw(m.vbo().count);
								}
								++idx;
							}
						}
						layouts.swap();
					}
				}
				flags.reset(Flag::eRecreated);
			}
			boot.device.waitIdle();
			text.mesh.reset();
			font.atlas.reset();
		}
	} catch (std::exception const& e) {
		logE("exception: {}", e.what());
	}

	engine::Service engine({argc, argv});
	std::array<io::Path, 2> const pathSearch = {"data", "demo/data"};
	auto dataPaths = engine::locate(pathSearch);
	engine::Info info;
	Window::Info windowInfo;
	windowInfo.config.size = {1280, 720};
	windowInfo.config.title = "LittleEngineVk Demo";
	info.windowInfo = std::move(windowInfo);
	info.dataPaths = dataPaths;
#if defined(LEVK_DEBUG)
// info.bLogVRAMallocations = true;
#endif
	if (!engine.init(std::move(info))) {
		return 1;
	}
	engine::g_shutdownSequence = engine::ShutdownSequence::eShutdown_CloseWindow;
	while (engine.running()) {
		engine.update(g_driver);
		engine.render();
	}
	g_driver.cleanup();
	return 0;
}
