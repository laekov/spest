#include <spest/tb_sim.h>
#include <spest/debug.h>
#include <spest/hw_spec.h>

#include <algorithm>

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
	cnt_t max_line = 0;
	cnt_t tot_trans = 0;
	for (auto& line : global_lds) {
		// std::cerr << lookupHashRecord(line.first) << std::endl;

		size_t maxlen = 0;
		for (auto& v : line.second) {
			maxlen = std::max(maxlen, v.size());
			tot_access += v.size();
		}
		max_line = std::max((cnt_t)maxlen, max_line);
		for (size_t i = 0; i < maxlen; ++i) {
			std::vector<unsigned long> addrs;
			for (auto& v : line.second) {
				if (v.size() > i) {
					addrs.push_back(v[i].addr);
				}
			}
			std::sort(addrs.begin(), addrs.end());
			if (*addrs.begin() == *addrs.rbegin()) {
				/*
				 * all the same memory visit
				 * should be merged (?)
				 */
				// tot_trans += gpu->getGlobalMemLat(num_threads, 4) * ((addrs.size() + 1) / 8);
				tot_trans += 1.; // / gpu->getGlobalMemLat(num_threads, 4);
				// std::cerr << lookupHashRecord(line.first) << " " << addrs[0] << " " << addrs.size() << "\n";
			} else {
				/*
				 * not all the same, should look for merge
				 */
				double local_sum = 0;
				for (size_t i = 0, j; i < addrs.size(); ) {
					for (j = i + 1; j < addrs.size() && addrs[j] == addrs[i]; ++j);
					if (j > i + 1) {
						// std::cerr << "SAME " << addrs[j] << " " << j - i << "\n";
						i = j;
						local_sum += (j - i) / gpu->getGlobalMemLat(num_threads, 4) / 1.5;
						continue;
					}
					for (j = i + 1; j < addrs.size() && addrs[j] == addrs[j - 1] + 4; ++j);
					local_sum += (j - i) / gpu->getGlobalMemLat(num_threads, (j - i));
					// std::cerr << "CONT " << addrs[j] << " " << j - i << "\n";
					i = j;
				}
				tot_trans += 1.; // local_sum / addrs.size();
			}
		}
	}
	// exit(0);
	// std::cout << tot_trans << "\n";
	return tot_trans;
}
