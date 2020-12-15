#include <map>
#include <graphics/context/physical_device.hpp>

namespace le::graphics {
bool PhysicalDevice::surfaceSupport(u32 queueFamily, vk::SurfaceKHR surface) const {
	return !default_v(device) && device.getSurfaceSupportKHR(queueFamily, surface);
}

vk::SurfaceCapabilitiesKHR PhysicalDevice::surfaceCapabilities(vk::SurfaceKHR surface) const {
	return !default_v(device) ? device.getSurfaceCapabilitiesKHR(surface) : vk::SurfaceCapabilitiesKHR();
}

PhysicalDevice DevicePicker::pick(Span<PhysicalDevice> devices) const {
	ENSURE(!devices.empty(), "No devices to pick from");
	using DevList = std::vector<Ref<PhysicalDevice const>>;
	std::map<Score, DevList, std::greater<Score>> deviceMap;
	for (auto const& device : devices) {
		deviceMap[score(device)].emplace_back(device);
	}
	DevList const& list = deviceMap.begin()->second;
	return list.size() == 1 ? list.front().get() : tieBreak(deviceMap.begin()->second);
}

DevicePicker::Score DevicePicker::score(PhysicalDevice const& device) const {
	Score total = 0;
	addIf(total, device.discreteGPU(), discrete);
	addIf(total, device.integratedGPU(), integrated);
	return modify(total, device);
}
} // namespace le::graphics
