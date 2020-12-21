#pragma once
#include <unordered_map>
#include <core/ref.hpp>
#include <core/span.hpp>
#include <core/view.hpp>
#include <graphics/utils/ring_buffer.hpp>
#include <graphics/utils/rw_span.hpp>
#include <vulkan/vulkan.hpp>

namespace le::graphics {
class Device;
class VRAM;
class Texture;
struct Buffer;
struct Image;

struct BindingInfo {
	vk::DescriptorSetLayoutBinding binding;
	std::string name;
	bool bUnassigned = false;
};

class DescriptorSet {
  public:
	// Combined Image Sampler
	struct CIS {
		vk::ImageView image;
		vk::Sampler sampler;
	};
	struct CreateInfo;

	DescriptorSet(VRAM& vram, CreateInfo const& info);
	DescriptorSet(DescriptorSet&&) noexcept;
	DescriptorSet& operator=(DescriptorSet&&) noexcept;
	~DescriptorSet();

	void index(std::size_t index);
	void next();
	vk::DescriptorSet get() const;
	std::vector<CView<Buffer>> buffers(u32 binding) const;

	template <typename T>
	bool writeBuffer(u32 binding, T const& data, vk::DescriptorType type = vk::DescriptorType::eUniformBuffer);
	template <typename T, typename Cont = Span<T>>
	bool writeBuffers(u32 binding, Cont&& data, vk::DescriptorType type = vk::DescriptorType::eUniformBuffer);
	bool writeBuffers(u32 binding, void* pData, std::size_t size, std::size_t count, vk::DescriptorType type = vk::DescriptorType::eUniformBuffer);
	void updateBuffers(u32 binding, Span<CView<Buffer>> buffers, std::size_t size, vk::DescriptorType type = vk::DescriptorType::eUniformBuffer);
	bool updateCIS(u32 binding, std::vector<CIS> cis);
	bool updateTextures(u32 binding, Span<Texture> textures);

	u32 setNumber() const noexcept;

  private:
	void destroy();
	View<Buffer> resize(View<Buffer> old, std::size_t size, vk::DescriptorType type, std::string_view name) const;

	struct Binding {
		std::string name;
		vk::DescriptorType type;
		std::vector<View<Buffer>> buffers;
		std::vector<CIS> cis;
		u32 count = 1;
	};
	struct Set {
		vk::DescriptorSet set;
		vk::DescriptorPool pool;
		std::unordered_map<u32, Binding> bindings;
	};
	struct Storage {
		vk::DescriptorSetLayout layout;
		RingBuffer<Set> setBuffer;
		std::unordered_map<u32, BindingInfo> bindingInfos;
		u32 rotateCount = 1;
		u32 setNumber = 0;
	} m_storage;
	Ref<VRAM> m_vram;
	Ref<Device> m_device;
};

struct DescriptorSet::CreateInfo {
	vk::DescriptorSetLayout layout;
	Span<BindingInfo> bindingInfos;
	std::size_t rotateCount = 1;
	u32 setNumber = 0;
};

class SetFactory {
  public:
	struct CreateInfo;

	SetFactory(VRAM& vram, CreateInfo const& info);

	DescriptorSet& front();
	DescriptorSet& at(std::size_t idx);
	RWSpan<DescriptorSet> populate(std::size_t count);
	void swap();

  private:
	struct Storage {
		vk::DescriptorSetLayout layout;
		std::vector<BindingInfo> bindInfos;
		std::vector<DescriptorSet> descriptorSets;
		std::size_t rotateCount = 0;
		u32 setNumber = 0;
	} m_storage;
	Ref<VRAM> m_vram;
	Ref<Device> m_device;
};

struct SetFactory::CreateInfo {
	vk::DescriptorSetLayout layout;
	std::vector<BindingInfo> bindInfos;
	std::size_t rotateCount = 2;
	u32 setNumber = 0;
};

// impl

inline u32 DescriptorSet::setNumber() const noexcept {
	return m_storage.setNumber;
}

template <typename T>
bool DescriptorSet::writeBuffer(u32 binding, T const& data, vk::DescriptorType type) {
	return writeBuffers(binding, (void*)&data, sizeof(T), 1, type);
}
template <typename T, typename Cont>
bool DescriptorSet::writeBuffers(u32 binding, Cont&& data, vk::DescriptorType type) {
	ENSURE(data.size() > 0, "Empty container");
	return writeBuffers(binding, (void*)std::addressof(*data.begin()), sizeof(T), data.size(), type);
}
} // namespace le::graphics
