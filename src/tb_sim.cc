#include <spest/tb_sim.h>
#include <spest/debug.h>
#include <spest/hw_spec.h>
#include <spest/shfl.h>

#include <cstring>
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

void TBSim::checkCaller(hash_t caller) {
	if (global_lds.find(caller) == global_lds.end()) {
		global_lds[caller].resize(this->sz.n());
	}
}

void TBSim::ld(void* addr, size_t sz, hash_t caller, TDim thid) {
	if (global_lds.find(caller) == global_lds.end()) {
		checkCaller(caller);
	}
	global_lds[caller][getTh(thid)].push_back(MemAccess(sz, addr));
}

void TBSim::ld(void* addr, size_t sz, hash_t caller, ShflOp* shfl, size_t scale, TDim thid) {
	if (global_lds.find(caller) == global_lds.end()) {
		checkCaller(caller);
	}
	MemAccess m(sz, addr, shfl);
	m.scale = scale;
	global_lds[caller][getTh(thid)].push_back(m);
}

void TBSim::shfl(ShflOp* op, TDim thid) {
	shfls[getTh(thid)].push_back(op);
}

int TBSim::getTh(TDim idx) {
	return idx.toid(sz);
}

int TBSim::resolveShfls() {
	int rest_shfl = 0, n = shfls.size();
	for (auto& s : shfls) {
		rest_shfl += s.size();
	}
	std::vector<int> si;
	si.resize(n);
	while (rest_shfl) {
		int first = -1, size = -1;
		for (int i = 0; i < n; ++i) {
			if (si[i] == shfls[i].size()) {
				continue;
			}
			int gran = shfls[i][si[i]]->gran;
			if (i / gran * gran != i) {
				continue;
			}
			bool can_process = true;
			for (int j = 0; j < gran; ++j) {
				if (i + j >= n) {
					return 1;
				}
				if (si[i + j] >= shfls[i + j].size()) {
					return 2;
				}
				if (shfls[i + j][si[i + j]]->gran != gran) {
					can_process = false;
					break;
				}
			}
			if (!can_process) {
				continue;
			}
			first = i, size = gran;
			break;
		}
		if (first == -1) {
			return 3;
		}
		for (int i = first; i < first + size; ++i) {
			int j = first + shfls[i][si[i]]->tgt_rank % size;
			memcpy(shfls[i][si[i]]->res, shfls[j][si[j]]->val, shfls[i][si[i]]->sz);
		}
		for (int i = first; i < first + size; ++i) {
			++si[i], --rest_shfl;
		}
	}
	shfls.resize(0);
	return 0;
}

cnt_t TBSim::calculate(int num_threads) {
	/* 
	 * Assume that all memory accesses are 4-byte
	 */
	if (auto res = resolveShfls()) {
		SPEST_LOG("Failed to resolve shufls with error code " << res);
	}
	auto gpu = HwSpec::getPlatform("system");
	cnt_t tot_access = 0;
	cnt_t est_lat = 0;
	for (auto& line : global_lds) {
		// std::cout << lookupHashRecord(line.first) << std::endl;

		size_t maxlen = 0;
		for (auto& v : line.second) {
			maxlen = std::max(maxlen, v.size());
		}

		tot_access += maxlen;

		for (size_t i = 0; i < maxlen; ++i) {
			std::vector<unsigned long> addrs;
			for (auto& v : line.second) {
				if (v.size() > i) {
					addrs.push_back(v[i].getAddr());
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
			// est_lat += line.second.size() * lat; // addrs.size() * lat;
			est_lat += addrs.size() * lat; // addrs.size() * lat;
			// std::cout << addrs.size() * lat << std::endl;
			// std::cout << same_size << " " << group_size << " " << num_threads << " " << lat << "\n";
		}
	}
	// std::cout << tot_access << " " << est_lat << "\n";
	// exit(0);
	return est_lat;
}
