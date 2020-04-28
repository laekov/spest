#include <spest/tb_sim.h>
#include <spest/debug.h>
#include <spest/hw_spec.h>

#include <algorithm>

int getMode(std::vector<int> a) {
	int modv = 0, modc = -1;
	std::sort(a.begin(), a.end());
	int n = a.size();
	for (int i = 0, j; i < n; ++i) {
		for (j = i; j < n && a[j] == a[i]; ++j);
		if (j - i > modc) {
			modc = j - i;
			modv = a[i];
		}
	}
	return modv;
}

void TBSim::ld(void* addr, size_t sz, hash_t caller) {
	if (global_lds.find(caller) == global_lds.end()) {
		global_lds[caller].resize(this->sz.n());
	}
	global_lds[caller][current_thread].push_back(MemAccess(sz, addr));
}

void TBSim::nextThread() {
	++current_thread;
}

cnt_t TBSim::calculate(int num_threads) {
	/* 
	 * Assume that all memory accesses are 4-byte
	 */
	auto gpu = HwSpec::getPlatform("vegavii");
	cnt_t tot_access = 0;
	cnt_t est_lat = 0;
	for (auto& line : global_lds) {
		// std::cerr << lookupHashRecord(line.first) << std::endl;

		size_t maxlen = 0;
		for (auto& v : line.second) {
			maxlen = std::max(maxlen, v.size());
		}

		tot_access += maxlen;

		for (size_t i = 0; i < maxlen; ++i) {
			std::vector<unsigned long> addrs;
			for (auto& v : line.second) {
				if (v.size() > i) {
					addrs.push_back(v[i].addr);
				}
			}
			std::sort(addrs.begin(), addrs.end());

			std::vector<int> same_sizes;
			for (size_t i = 0, j; i < addrs.size(); i = j) {
				for (j = i + 1; j < addrs.size() && addrs[j] == addrs[i]; ++j);
				same_sizes.push_back(j - i);
			}
			int same_size = getMode(same_sizes);

			size_t last_addr = std::unique(addrs.begin(), addrs.end()) - addrs.begin();
			std::vector<int> group_szs;
			for (size_t i = 0, j; i < last_addr; i = j) {
				for (j = i + 1; j < addrs.size() && addrs[j] == addrs[j - 1] + 4; ++j);
				group_szs.push_back((j - i) * sizeof(int));
			}
			int group_size = getMode(group_szs);

			auto lat = gpu->getGlobalMemBw(num_threads, group_size, same_size);
			est_lat += addrs.size() * lat;
			// std::cout << addrs.size() << " " << lat << "\n";
		}
	}
	// exit(0);
	// std::cout << tot_access << " " << est_lat << "\n";
	// exit(0);
	return est_lat;
}
