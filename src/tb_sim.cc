#include <spest/tb_sim.h>
#include <spest/debug.h>
#include <spest/hw_spec.h>

#include <algorithm>

void TBSim::ld(void* addr, size_t sz) {
	global_lds[current_thread].push_back(MemAccess(sz, addr));
}

void TBSim::nextThread() {
	++current_thread;
}

cnt_t TBSim::calculate(int num_threads) {
	/* 
	 * Assume that all memory accesses are 4-byte
	 */
	std::vector<unsigned long> addrs;
	for (auto& v : global_lds) {
		for (auto& a : v) {
			addrs.push_back(a.addr);
		}
	}
	auto gpu = HwSpec::getPlatform("vegavii");
	std::sort(addrs.begin(), addrs.end());
	cnt_t tot_trans = 0;
	for (size_t i = 0, j; i < addrs.size(); ) {
		for (j = i + 1; j < addrs.size() && addrs[j] == addrs[i]; ++j);
		if (j > i + 1) {
			i = j;
			continue;
		}
		for (j = i + 1; j < addrs.size() && addrs[j] == addrs[j - 1] + 4 && j - i < 512; ++j);
		tot_trans += gpu->getGlobalMemLat(num_threads, (j - i) / 4);
		i = j;
	}
	return tot_trans;
}
