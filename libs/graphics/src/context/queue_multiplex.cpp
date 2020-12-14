#include <map>
#include <set>
#include <graphics/context/device.hpp>
#include <graphics/context/instance.hpp>
#include <graphics/context/queue_multiplex.hpp>

namespace le::graphics {
namespace {
using vkqf = vk::QueueFlagBits;

class Selector {
  public:
	Selector(std::vector<QueueFamily> families) : m_families(std::move(families)) {
		QFlags found;
		for (auto const& family : m_families) {
			found.set(family.flags);
		}
		bool const valid = found.all(QFlags::inverse());
		ENSURE(valid, "Required queues not present");
		logE_if(!valid, "[{}] Required Vulkan Queues not present on selected physical device!");
	}

	QueueFamily* exact(QFlags flags) {
		for (auto& f : m_families) {
			// Only return if queue flags match exactly
			if (f.flags == flags && f.reserved < f.total) {
				return &f;
			}
		}
		return nullptr;
	}

	QueueFamily* best(QFlags flags) {
		for (auto& f : m_families) {
			// Return if queue supports desired flags
			if (f.flags.test(flags) && f.reserved < f.total) {
				return &f;
			}
		}
		return nullptr;
	}

	template <typename T>
	using stdil = std::initializer_list<T>;

	QueueFamily* reserve(stdil<QFlags> combo) {
		QueueFamily* f = nullptr;
		// First pass: exact match
		for (QFlags flags : combo) {
			if (f = exact(flags); f && f->reserved < f->total) {
				break;
			}
		}
		if (!f || f->reserved >= f->total) {
			// Second pass: best match
			for (QFlags flags : combo) {
				if (f = best(flags); f && f->reserved < f->total) {
					break;
				}
			}
		}
		if (f && f->reserved < f->total) {
			// Match found, reserve queue and return
			++f->reserved;
			return f;
		}
		// No matches found / no queues left to reserve
		return nullptr;
	}

	std::vector<QueueFamily> m_families;
	u32 m_queueCount = 0;
};

QueueMultiplex::QCI createInfo(QueueFamily& out_family, Span<f32> prio) {
	QueueMultiplex::QCI ret;
	ret.first.queueFamilyIndex = out_family.familyIndex;
	ret.first.queueCount = (u32)prio.size();
	ret.first.pQueuePriorities = prio.data();
	for (std::size_t i = 0; i < prio.size(); ++i) {
		QueueMultiplex::Queue queue;
		ENSURE(out_family.nextQueueIndex < out_family.total, "No queues remaining");
		queue.arrayIndex = out_family.nextQueueIndex++;
		queue.familyIndex = out_family.familyIndex;
		ret.second.push_back(queue);
	}
	return ret;
}

template <typename T, typename... Ts>
constexpr bool uniqueFam(T&& t, Ts&&... ts) noexcept {
	return ((t.familyIndex != ts.familyIndex) && ...);
}
template <typename T, typename... Ts>
constexpr bool uniqueQueue(T&& t, Ts&&... ts) noexcept {
	return ((t.familyIndex != ts.familyIndex && t.arrayIndex != ts.arrayIndex) && ...);
}
} // namespace

std::vector<vk::DeviceQueueCreateInfo> QueueMultiplex::select(std::vector<QueueFamily> families) {
	std::vector<vk::DeviceQueueCreateInfo> ret;
	Selector sl(std::move(families));
	// Reserve one queue for graphics/present
	auto fpg = sl.reserve({QType::eGraphics | QType::ePresent});
	// Reserve another for transfer
	auto ft = sl.reserve({QType::eTransfer, QType::eTransfer | QType::ePresent, QType::eTransfer | QType::eGraphics});
	// Can't function without graphics/present
	if (!fpg) {
		return ret;
	}
	if (ft && uniqueFam(*fpg, *ft)) {
		// Two families, two queues
		static std::array const prio = {1.0f};
		makeQueues(ret, makeFrom2(*fpg, *ft, prio, prio), {{{0, 0}, {0, 0}, {1, 0}}});
	} else {
		if (fpg->total > 1) {
			// One family, two queues
			static std::array const prio = {0.8f, 0.2f};
			makeQueues(ret, makeFrom1(*fpg, prio), {{{0, 0}, {0, 0}, {0, 1}}});
		} else {
			// One family, one queue
			static std::array const prio = {1.0f};
			makeQueues(ret, makeFrom1(*fpg, prio), {{{0, 0}, {0, 0}, {0, 0}}});
		}
	}
	m_queues[(std::size_t)QType::eGraphics].second = &m_mutexes.gp;
	m_queues[(std::size_t)QType::ePresent].second = &m_mutexes.gp;
	m_queues[(std::size_t)QType::eTransfer].second = &m_mutexes.t;
	return ret;
}

void QueueMultiplex::setup(vk::Device device) {
	std::set<u32> families, queues;
	for (auto& [queue, _] : m_queues) {
		queue.queue = device.getQueue(queue.familyIndex, queue.arrayIndex);
		families.insert(queue.familyIndex);
		queues.insert((queue.familyIndex << 4) ^ queue.arrayIndex);
	}
	m_familyCount = (u32)families.size();
	m_queueCount = (u32)queues.size();
	logD("[{}] Multiplexing [{}] Vulkan queue(s) from [{}] queue families for [Graphics/Present, Transfer]", g_name, m_queueCount, m_familyCount);
}

std::vector<u32> QueueMultiplex::familyIndices(QFlags flags) const {
	std::vector<u32> ret;
	ret.reserve(3);
	if (flags.test(QType::eGraphics)) {
		ret.push_back(queue(QType::eGraphics).familyIndex);
	}
	if (flags.test(QType::ePresent) && queue(QType::ePresent).familyIndex != queue(QType::eGraphics).familyIndex) {
		ret.push_back(queue(QType::ePresent).familyIndex);
	}
	if (flags.test(QType::eTransfer) && queue(QType::eTransfer).familyIndex != queue(QType::eGraphics).familyIndex) {
		ret.push_back(queue(QType::eTransfer).familyIndex);
	}
	return ret;
}

vk::Result QueueMultiplex::present(vk::PresentInfoKHR const& info, bool bLock) {
	auto& q = queue(QType::ePresent);
	if (!bLock) {
		return q.queue.presentKHR(info);
	} else {
		auto lock = mutex(QType::ePresent).lock();
		return q.queue.presentKHR(info);
	}
}

void QueueMultiplex::submit(QType type, vAP<vk::SubmitInfo> infos, vk::Fence signal, bool bLock) {
	auto& q = queue(type);
	if (!bLock) {
		q.queue.submit(infos, signal);
	} else {
		auto lock = mutex(QType::ePresent).lock();
		q.queue.submit(infos, signal);
	}
}

QueueMultiplex::QCIArr<1> QueueMultiplex::makeFrom1(QueueFamily& gpt, Span<f32> prio) {
	return {createInfo(gpt, prio)};
}

QueueMultiplex::QCIArr<2> QueueMultiplex::makeFrom2(QueueFamily& a, QueueFamily& b, Span<f32> pa, Span<f32> pb) {
	std::array<QueueMultiplex::QCI, 2> ret;
	ret[0] = createInfo(a, pa);
	ret[1] = createInfo(b, pb);
	return ret;
}

QueueMultiplex::QCIArr<3> QueueMultiplex::makeFrom3(QueueFamily& g, QueueFamily& p, QueueFamily& t, Span<f32> pg, Span<f32> pp, Span<f32> pt) {
	std::array<QueueMultiplex::QCI, 3> ret;
	ret[0] = createInfo(g, pg);
	ret[1] = createInfo(p, pp);
	ret[2] = createInfo(t, pt);
	return ret;
}

void QueueMultiplex::makeQueues(qcivec& out_vec, Span<QCI> qcis, Assign const& a) {
	for (auto const& [info, _] : qcis) {
		out_vec.push_back(info);
	}
	assign(qcis[a[0].first].second[a[0].second], qcis[a[1].first].second[a[1].second], qcis[a[2].first].second[a[2].second]);
}

void QueueMultiplex::assign(Queue g, Queue p, Queue t) {
	if (uniqueQueue(g, p, t)) {
		g.bUnique = p.bUnique = t.bUnique = true;
	} else if (uniqueQueue(g, p)) {
		g.bUnique = p.bUnique = true;
	} else if (uniqueQueue(g, t)) {
		g.bUnique = t.bUnique = true;
	} else if (uniqueQueue(p, t)) {
		p.bUnique = t.bUnique = true;
	}
	queue(QType::eGraphics) = g;
	queue(QType::ePresent) = p;
	queue(QType::eTransfer) = t;
}
} // namespace le::graphics