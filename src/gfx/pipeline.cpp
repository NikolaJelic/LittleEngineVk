#include <array>
#include <fmt/format.h>
#include <core/log.hpp>
#include <engine/gfx/pipeline.hpp>
#include <engine/resources/resources.hpp>
#include <gfx/deferred.hpp>
#include <gfx/device.hpp>
#include <gfx/pipeline_impl.hpp>
#include <gfx/render_context.hpp>
#include <gfx/resource_descriptors.hpp>
#include <resources/resources_impl.hpp>

namespace le::gfx {
PipelineImpl::PipelineImpl() = default;
PipelineImpl::PipelineImpl(PipelineImpl&&) = default;
PipelineImpl& PipelineImpl::operator=(PipelineImpl&&) = default;

PipelineImpl::~PipelineImpl() {
	destroy();
}

bool PipelineImpl::create(Info info) {
	m_info = std::move(info);
	if (m_info.shaderID.empty()) {
		m_info.shaderID = "shaders/default";
	}
	if (create()) {
#if defined(LEVK_RESOURCES_HOT_RELOAD)
		if (auto pImpl = res::impl(m_info.shader)) {
			m_reloadToken = pImpl->onReload.subscribe([this]() { m_bShaderReloaded = true; });
		}
#endif
		return true;
	}
	return false;
}

bool PipelineImpl::update(RenderPass const& renderPass) {
	bool bOutOfDate = renderPass.renderPass != vk::RenderPass() && renderPass.renderPass != m_info.renderPass;
#if defined(LEVK_RESOURCES_HOT_RELOAD)
	bOutOfDate |= m_bShaderReloaded;
	m_bShaderReloaded = false;
#endif
	if (bOutOfDate) {
		// Add a frame of padding since this frame hasn't completed drawing yet
		deferred::release([pipeline = m_pipeline, layout = m_layout]() { g_device.destroy(pipeline, layout); }, 1);
		m_info.renderPass = renderPass.renderPass;
		return create();
	}
	return true;
}

void PipelineImpl::destroy() {
	deferred::release(m_pipeline, m_layout);
	m_pipeline = vk::Pipeline();
	m_layout = vk::PipelineLayout();
	return;
}

bool PipelineImpl::create() {
	if ((m_info.shader.guid == res::GUID::null || m_info.shader.status() != res::Status::eReady) && !m_info.shaderID.empty()) {
		if (auto shader = res::find<res::Shader>(m_info.shaderID)) {
			m_info.shader = *shader;
		}
	}
	ENSURE(m_info.shader.status() == res::Status::eReady, "Shader is not ready!");
	auto pShaderImpl = res::impl(m_info.shader);
	if (m_info.shader.status() != res::Status::eReady || !pShaderImpl) {
		logE("Failed to create pipeline!");
		return false;
	}
	std::array const setLayouts = {rd::g_bufferLayout, rd::g_samplerLayout};
	m_layout = g_device.createPipelineLayout(m_info.pushConstantRanges, setLayouts);
	vk::PipelineVertexInputStateCreateInfo vertexInputState;
	{
		vertexInputState.vertexBindingDescriptionCount = (u32)m_info.vertexBindings.size();
		vertexInputState.pVertexBindingDescriptions = m_info.vertexBindings.data();
		vertexInputState.vertexAttributeDescriptionCount = (u32)m_info.vertexAttributes.size();
		vertexInputState.pVertexAttributeDescriptions = m_info.vertexAttributes.data();
	}
	vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState;
	{
		inputAssemblyState.topology = vk::PrimitiveTopology::eTriangleList;
		inputAssemblyState.primitiveRestartEnable = false;
	}
	vk::PipelineViewportStateCreateInfo viewportState;
	{
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;
	}
	vk::PipelineRasterizationStateCreateInfo rasterizerState;
	{
		rasterizerState.depthClampEnable = false;
		rasterizerState.rasterizerDiscardEnable = false;
		rasterizerState.polygonMode = m_info.polygonMode;
		rasterizerState.lineWidth = m_info.staticLineWidth;
		rasterizerState.cullMode = m_info.cullMode;
		rasterizerState.frontFace = m_info.frontFace;
		rasterizerState.depthBiasEnable = false;
	}
	vk::PipelineMultisampleStateCreateInfo multisamplerState;
	{
		multisamplerState.sampleShadingEnable = false;
		multisamplerState.rasterizationSamples = vk::SampleCountFlagBits::e1;
	}
	vk::PipelineColorBlendAttachmentState colorBlendAttachment;
	{
		colorBlendAttachment.colorWriteMask = m_info.colourWriteMask;
		colorBlendAttachment.blendEnable = m_info.flags.test(Pipeline::Flag::eBlend);
		colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
		colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
		colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
		colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
		colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;
	}
	vk::PipelineColorBlendStateCreateInfo colorBlendState;
	{
		colorBlendState.logicOpEnable = false;
		colorBlendState.attachmentCount = 1;
		colorBlendState.pAttachments = &colorBlendAttachment;
	}
	vk::PipelineDepthStencilStateCreateInfo depthStencilState;
	{
		depthStencilState.depthTestEnable = m_info.flags.test(Pipeline::Flag::eDepthTest);
		depthStencilState.depthWriteEnable = m_info.flags.test(Pipeline::Flag::eDepthWrite);
		depthStencilState.depthCompareOp = vk::CompareOp::eLess;
	}
	auto states = m_info.dynamicStates;
	states.insert(vk::DynamicState::eViewport);
	states.insert(vk::DynamicState::eScissor);
	std::vector<vk::DynamicState> stateFlags = {states.begin(), states.end()};
	vk::PipelineDynamicStateCreateInfo dynamicState;
	{
		dynamicState.dynamicStateCount = (u32)stateFlags.size();
		dynamicState.pDynamicStates = stateFlags.data();
	}
	std::vector<vk::PipelineShaderStageCreateInfo> shaderCreateInfo;
	{
		auto modules = pShaderImpl->modules();
		ENSURE(!modules.empty(), "No shader modules!");
		shaderCreateInfo.reserve(modules.size());
		for (auto const& [type, module] : modules) {
			vk::PipelineShaderStageCreateInfo createInfo;
			createInfo.stage = res::Shader::Impl::s_typeToFlagBit[(std::size_t)type];
			createInfo.module = module;
			createInfo.pName = "main";
			shaderCreateInfo.push_back(std::move(createInfo));
		}
	}
	vk::GraphicsPipelineCreateInfo createInfo;
	createInfo.stageCount = (u32)shaderCreateInfo.size();
	createInfo.pStages = shaderCreateInfo.data();
	createInfo.pVertexInputState = &vertexInputState;
	createInfo.pInputAssemblyState = &inputAssemblyState;
	createInfo.pViewportState = &viewportState;
	createInfo.pRasterizationState = &rasterizerState;
	createInfo.pMultisampleState = &multisamplerState;
	createInfo.pDepthStencilState = &depthStencilState;
	createInfo.pColorBlendState = &colorBlendState;
	createInfo.pDynamicState = &dynamicState;
	createInfo.layout = m_layout;
	createInfo.renderPass = m_info.renderPass;
	createInfo.subpass = 0;
#if VK_HEADER_VERSION >= 131
	auto pipeline = g_device.device.createGraphicsPipeline({}, createInfo);
#else
	auto [result, pipeline] = g_device.device.createGraphicsPipeline({}, createInfo);
	if (result != vk::Result::eSuccess) {
		return false;
	}
#endif
	m_pipeline = pipeline;
#if defined(LEVK_RESOURCES_HOT_RELOAD)
	m_bShaderReloaded = false;
#endif
	return true;
}

namespace {
std::unordered_map<std::size_t, PipelineImpl> g_implMap;

std::size_t pipeHash(Pipeline const& pipe, vk::Format colour, vk::Format depth) {
	std::size_t hash = 0;
	hash ^= pipe.shader.guid;
	hash ^= (std::size_t)pipe.lineWidth;
	hash ^= pipe.flags.bits.to_ulong();
	hash ^= (std::size_t)pipe.cullMode;
	hash ^= (std::size_t)pipe.polygonMode;
	hash ^= std::hash<vk::Format>()(colour);
	hash ^= std::hash<vk::Format>()(depth);
	return hash;
}
} // namespace

PipelineImpl& pipes::find(Pipeline const& pipe, RenderPass const& renderPass) {
	auto const hash = pipeHash(pipe, renderPass.colour, renderPass.depth);
	auto search = g_implMap.find(hash);
	if (search != g_implMap.end()) {
		search->second.update(renderPass);
		return search->second;
	}
	auto& ret = g_implMap[hash];
	PipelineImpl::Info implInfo;
	implInfo.vertexBindings = rd::vbo::vertexBindings();
	implInfo.vertexAttributes = rd::vbo::vertexAttributes();
	implInfo.pushConstantRanges = rd::PushConstants::ranges();
	implInfo.renderPass = renderPass.renderPass;
	implInfo.polygonMode = (vk::PolygonMode)pipe.polygonMode;
	implInfo.cullMode = (vk::CullModeFlagBits)pipe.cullMode;
	implInfo.frontFace = (vk::FrontFace)pipe.frontFace;
	implInfo.staticLineWidth = pipe.lineWidth;
	implInfo.shader = pipe.shader;
	implInfo.flags = pipe.flags;
	ret.create(implInfo);
	return ret;
}

void pipes::deinit() {
	for (auto& [_, pipeline] : g_implMap) {
		pipeline.destroy();
	}
	g_implMap.clear();
}
} // namespace le::gfx
