#include "core/log.hpp"
#include "device.hpp"
#include "vram.hpp"
#include "resource_descriptors.hpp"
#include "engine/assets/resources.hpp"
#include "engine/gfx/texture.hpp"

namespace le::gfx::rd
{
View::View() = default;

View::View(Renderer::View const& view, u32 dirLightCount)
	: mat_vp(view.mat_vp), mat_v(view.mat_v), mat_p(view.mat_p), mat_ui(view.mat_ui), pos_v(view.pos_v), dirLightCount(dirLightCount)
{
}

Materials::Mat::Mat(Material const& material, Colour dropColour)
	: ambient(material.m_albedo.ambient.toVec4()),
	  diffuse(material.m_albedo.diffuse.toVec4()),
	  specular(material.m_albedo.specular.toVec4()),
	  dropColour(dropColour.toVec4()),
	  shininess(material.m_shininess)
{
}

DirLights::Light::Light(DirLight const& dirLight)
	: ambient(dirLight.ambient.toVec4()),
	  diffuse(dirLight.diffuse.toVec4()),
	  specular(dirLight.specular.toVec4()),
	  direction(dirLight.direction)
{
}

u32 ImageSamplers::s_max = 1024;

vk::DescriptorSetLayoutBinding const View::s_setLayoutBinding =
	vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vkFlags::vertFragShader);

vk::DescriptorSetLayoutBinding const Models::s_setLayoutBinding =
	vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vkFlags::vertFragShader);

vk::DescriptorSetLayoutBinding const Normals::s_setLayoutBinding =
	vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vkFlags::vertFragShader);

vk::DescriptorSetLayoutBinding const Materials::s_setLayoutBinding =
	vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, vkFlags::vertFragShader);

vk::DescriptorSetLayoutBinding const Tints::s_setLayoutBinding =
	vk::DescriptorSetLayoutBinding(4, vk::DescriptorType::eStorageBuffer, 1, vkFlags::vertFragShader);

vk::DescriptorSetLayoutBinding const Flags::s_setLayoutBinding =
	vk::DescriptorSetLayoutBinding(5, vk::DescriptorType::eStorageBuffer, 1, vkFlags::vertFragShader);

vk::DescriptorSetLayoutBinding const DirLights::s_setLayoutBinding =
	vk::DescriptorSetLayoutBinding(6, vk::DescriptorType::eStorageBuffer, 1, vkFlags::vertFragShader);

vk::DescriptorSetLayoutBinding ImageSamplers::s_diffuseLayoutBinding =
	vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eCombinedImageSampler, s_max, vkFlags::fragShader);

vk::DescriptorSetLayoutBinding ImageSamplers::s_specularLayoutBinding =
	vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, s_max, vkFlags::fragShader);

vk::DescriptorSetLayoutBinding const ImageSamplers::s_cubemapLayoutBinding =
	vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eCombinedImageSampler, 1, vkFlags::fragShader);

u32 ImageSamplers::total()
{
	return s_diffuseLayoutBinding.descriptorCount + s_specularLayoutBinding.descriptorCount + s_cubemapLayoutBinding.descriptorCount;
}

void ImageSamplers::clampDiffSpecCount(u32 hardwareMax)
{
	s_max = std::min(s_max, (hardwareMax - 1) / 2); // (total - cubemap) / (diffuse + specular)
	s_diffuseLayoutBinding.descriptorCount = s_specularLayoutBinding.descriptorCount = s_max;
}

void Writer::write(vk::DescriptorSet set, Buffer const& buffer) const
{
	vk::DescriptorBufferInfo bufferInfo;
	bufferInfo.buffer = buffer.buffer;
	bufferInfo.offset = 0;
	bufferInfo.range = buffer.writeSize;
	vk::WriteDescriptorSet descWrite;
	descWrite.dstSet = set;
	descWrite.dstBinding = binding;
	descWrite.dstArrayElement = 0;
	descWrite.descriptorType = type;
	descWrite.descriptorCount = 1;
	descWrite.pBufferInfo = &bufferInfo;
	g_device.device.updateDescriptorSets(descWrite, {});
	return;
}

void Writer::write(vk::DescriptorSet set, std::vector<TextureImpl const*> const& textures) const
{
	std::vector<vk::DescriptorImageInfo> imageInfos;
	imageInfos.reserve(textures.size());
	for (auto pTex : textures)
	{
		vk::DescriptorImageInfo imageInfo;
		imageInfo.imageView = pTex->imageView;
		imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		imageInfo.sampler = pTex->sampler;
		imageInfos.push_back(imageInfo);
	}
	vk::WriteDescriptorSet descWrite;
	descWrite.dstSet = set;
	descWrite.dstBinding = binding;
	descWrite.dstArrayElement = 0;
	descWrite.descriptorType = type;
	descWrite.descriptorCount = (u32)imageInfos.size();
	descWrite.pImageInfo = imageInfos.data();
	g_device.device.updateDescriptorSets(descWrite, {});
	return;
}

void Descriptor<ImageSamplers>::writeArray(std::vector<TextureImpl const*> const& textures, vk::DescriptorSet set) const
{
	m_writer.write(set, textures);
}

Set::Set() : m_view(vk::BufferUsageFlagBits::eUniformBuffer)
{
	m_diffuse.m_writer.binding = ImageSamplers::s_diffuseLayoutBinding.binding;
	m_diffuse.m_writer.type = ImageSamplers::s_diffuseLayoutBinding.descriptorType;
	m_specular.m_writer.binding = ImageSamplers::s_specularLayoutBinding.binding;
	m_specular.m_writer.type = ImageSamplers::s_diffuseLayoutBinding.descriptorType;
	m_cubemap.m_writer.binding = ImageSamplers::s_cubemapLayoutBinding.binding;
	m_cubemap.m_writer.type = ImageSamplers::s_cubemapLayoutBinding.descriptorType;
}

void Set::destroy()
{
	m_view.release();
	m_models.release();
	m_normals.release();
	m_materials.release();
	m_tints.release();
	m_flags.release();
	m_dirLights.release();
	return;
}

void Set::resetTextures(SamplerCounts const& counts)
{
	std::deque<Texture const*> const diffuse((size_t)counts.diffuse, Resources::inst().get<Texture>("textures/white"));
	std::deque<Texture const*> const specular((size_t)counts.specular, Resources::inst().get<Texture>("textures/black"));
	writeDiffuse(diffuse);
	writeSpecular(specular);
}

void Set::writeView(View const& view)
{
	m_view.writeValue(view, m_bufferSet);
	return;
}

void Set::initSSBOs()
{
	StorageBuffers ssbos;
	ssbos.models.ssbo.push_back({});
	ssbos.normals.ssbo.push_back({});
	ssbos.materials.ssbo.push_back({});
	ssbos.tints.ssbo.push_back({});
	ssbos.flags.ssbo.push_back({});
	ssbos.dirLights.ssbo.push_back({});
	writeSSBOs(ssbos);
}

void Set::writeSSBOs(StorageBuffers const& ssbos)
{
	ASSERT(!ssbos.models.ssbo.empty() && !ssbos.normals.ssbo.empty() && !ssbos.materials.ssbo.empty() && !ssbos.tints.ssbo.empty()
			   && !ssbos.flags.ssbo.empty(),
		   "Empty SSBOs!");
	m_models.writeArray(ssbos.models.ssbo, m_bufferSet);
	m_normals.writeArray(ssbos.normals.ssbo, m_bufferSet);
	m_materials.writeArray(ssbos.materials.ssbo, m_bufferSet);
	m_tints.writeArray(ssbos.tints.ssbo, m_bufferSet);
	m_flags.writeArray(ssbos.flags.ssbo, m_bufferSet);
	if (!ssbos.dirLights.ssbo.empty())
	{
		m_dirLights.writeArray(ssbos.dirLights.ssbo, m_bufferSet);
	}
	return;
}

void Set::writeDiffuse(std::deque<Texture const*> const& diffuse)
{
	std::vector<TextureImpl const*> diffuseImpl;
	diffuseImpl.reserve(diffuse.size());
	for (auto pTex : diffuse)
	{
		diffuseImpl.push_back(pTex->m_uImpl.get());
	}
	m_diffuse.writeArray(diffuseImpl, m_samplerSet);
	return;
}

void Set::writeSpecular(std::deque<Texture const*> const& specular)
{
	std::vector<TextureImpl const*> specularImpl;
	specularImpl.reserve(specular.size());
	for (auto pTex : specular)
	{
		specularImpl.push_back(pTex->m_uImpl.get());
	}
	m_specular.writeArray(specularImpl, m_samplerSet);
	return;
}

void Set::writeCubemap(Cubemap const& cubemap)
{
	m_cubemap.writeArray({cubemap.m_uImpl.get()}, m_samplerSet);
	return;
}

vk::DescriptorSetLayout createSamplerLayout(u32 diffuse, u32 specular)
{
	auto diffuseBinding = ImageSamplers::s_diffuseLayoutBinding;
	diffuseBinding.descriptorCount = diffuse;
	auto specularBinding = ImageSamplers::s_specularLayoutBinding;
	specularBinding.descriptorCount = specular;
	std::array const textureBindings = {diffuseBinding, specularBinding, ImageSamplers::s_cubemapLayoutBinding};
	vk::DescriptorSetLayoutCreateInfo samplerLayoutInfo;
	samplerLayoutInfo.bindingCount = (u32)textureBindings.size();
	samplerLayoutInfo.pBindings = textureBindings.data();
	return g_device.device.createDescriptorSetLayout(samplerLayoutInfo);
}

SetLayouts allocateSets(u32 copies, SamplerCounts const& samplerCounts)
{
	SetLayouts ret;
	auto diffuseBinding = ImageSamplers::s_diffuseLayoutBinding;
	diffuseBinding.descriptorCount = samplerCounts.diffuse;
	auto specularBinding = ImageSamplers::s_specularLayoutBinding;
	specularBinding.descriptorCount = samplerCounts.specular;
	std::array const textureBindings = {diffuseBinding, specularBinding, ImageSamplers::s_cubemapLayoutBinding};
	vk::DescriptorSetLayoutCreateInfo samplerLayoutInfo;
	samplerLayoutInfo.bindingCount = (u32)textureBindings.size();
	samplerLayoutInfo.pBindings = textureBindings.data();
	ret.samplerLayout = g_device.device.createDescriptorSetLayout(samplerLayoutInfo);
	ret.sets.reserve((size_t)copies);
	for (u32 idx = 0; idx < copies; ++idx)
	{
		Set set;
		// Pool of descriptors
		vk::DescriptorPoolSize uboPoolSize;
		uboPoolSize.type = vk::DescriptorType::eUniformBuffer;
		uboPoolSize.descriptorCount = View::s_setLayoutBinding.descriptorCount;
		vk::DescriptorPoolSize ssboPoolSize;
		ssboPoolSize.type = vk::DescriptorType::eStorageBuffer;
		ssboPoolSize.descriptorCount = 6; // 6 members per SSBO
		vk::DescriptorPoolSize samplerPoolSize;
		samplerPoolSize.type = vk::DescriptorType::eCombinedImageSampler;
		samplerPoolSize.descriptorCount = ImageSamplers::total();
		std::array const bufferPoolSizes = {uboPoolSize, ssboPoolSize, samplerPoolSize};
		std::array const samplerPoolSizes = {samplerPoolSize};
		vk::DescriptorPoolCreateInfo createInfo;
		createInfo.poolSizeCount = (u32)bufferPoolSizes.size();
		createInfo.pPoolSizes = bufferPoolSizes.data();
		createInfo.maxSets = 1;
		set.m_bufferPool = g_device.device.createDescriptorPool(createInfo);
		createInfo.poolSizeCount = (u32)samplerPoolSizes.size();
		createInfo.pPoolSizes = samplerPoolSizes.data();
		set.m_samplerPool = g_device.device.createDescriptorPool(createInfo);
		// Allocate sets
		vk::DescriptorSetAllocateInfo allocInfo;
		allocInfo.descriptorPool = set.m_bufferPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &g_bufferLayout;
		auto const bufferSets = g_device.device.allocateDescriptorSets(allocInfo);
		allocInfo.descriptorPool = set.m_samplerPool;
		allocInfo.pSetLayouts = &ret.samplerLayout;
		auto const samplerSets = g_device.device.allocateDescriptorSets(allocInfo);
		// Write handles
		set.m_bufferSet = bufferSets.front();
		set.m_samplerSet = samplerSets.front();
		set.writeView({});
		set.initSSBOs();
		set.resetTextures(samplerCounts);
		ret.sets.push_back(std::move(set));
	}
	return ret;
}

void init()
{
	if (g_bufferLayout == vk::DescriptorSetLayout())
	{
		std::array const bufferBindings = {View::s_setLayoutBinding,	  Models::s_setLayoutBinding, Normals::s_setLayoutBinding,
										   Materials::s_setLayoutBinding, Tints::s_setLayoutBinding,  Flags::s_setLayoutBinding,
										   DirLights::s_setLayoutBinding};
		vk::DescriptorSetLayoutCreateInfo bufferLayoutInfo;
		bufferLayoutInfo.bindingCount = (u32)bufferBindings.size();
		bufferLayoutInfo.pBindings = bufferBindings.data();
		rd::g_bufferLayout = g_device.device.createDescriptorSetLayout(bufferLayoutInfo);
	}
	return;
}

void deinit()
{
	if (g_bufferLayout != vk::DescriptorSetLayout())
	{
		g_device.destroy(g_bufferLayout);
		g_bufferLayout = vk::DescriptorSetLayout();
	}
	return;
}
} // namespace le::gfx::rd
