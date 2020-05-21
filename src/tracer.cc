#include <omp.h>

#include <spest/tracer.h>
#include <spest/debug.h>


void Tracer::sim(TDim blocks, TDim threads) {
	sz = blocks;
	shp = threads;
	cusim = new CUSim(blocks, threads);
}

void Tracer::limitTBperCU(int n) {
	cusim->max_tb = std::min(cusim->max_tb, n);
}

struct LdRecord {
	void* addr;
	size_t sz;
	hash_t caller; 
	ShflOp* shfl;
	size_t scale;
};

struct ShflRecord {
	ShflOp* shfl;
};

std::vector<std::vector<LdRecord> > local_lds;
std::vector<std::vector<ShflRecord>  > local_shfls;


void Tracer::registerThread(int omp_thread_idx, TDim blockIdx, TDim threadIdx) {
	if (block_idxs.size() == 0) {
#pragma omp critical
		if (block_idxs.size() == 0) {
			auto nth = omp_get_num_threads();
			block_idxs.resize(nth);
			thread_idxs.resize(nth);
			local_lds.resize(sz.n() * shp.n());
			local_shfls.resize(sz.n() * shp.n());
		}
	}
	block_idxs[omp_thread_idx] = blockIdx;
	thread_idxs[omp_thread_idx] = threadIdx;
}

unsigned long Tracer::getIdxByThread() {
	auto thread_idx = omp_get_thread_num();
	auto data_idx = block_idxs[thread_idx].toid(sz) * shp.n() + 
		thread_idxs[thread_idx].toid(shp);
	return data_idx;
}

void Tracer::ld(void* addr, size_t sz, hash_t caller) {
	auto idx = getIdxByThread();
	local_lds[idx].push_back({
			.addr = addr, .sz = sz, .caller = caller, .shfl = 0
	});
	// getTBSim()->ld(addr, sz, caller, getTh());
}

void Tracer::ld(void* addr, size_t sz, hash_t caller, ShflOp* shfl, size_t scale) {
	auto idx = getIdxByThread();
	local_lds[idx].push_back({
			.addr = addr, sz = sz, caller = caller, .shfl = shfl, .scale = scale
	});
	// getTBSim()->ld(addr, sz, caller, shfl, scale, getTh());
}

void Tracer::shfl(ShflOp* op) {
	local_shfls[omp_get_thread_num()].push_back({
			.shfl = op
	});
    // getTBSim()->shfl(op, getTh());
}

cnt_t Tracer::get() const {
	size_t n_tbs = cusim->tbs.size();
#pragma omp parallel for schedule(dynamic, 4)
	for (size_t i = 0; i < n_tbs; ++i) {
		auto tb_sim = cusim->tbs[i];
		ENUM_TDIM(threadIdx, shp) {
			EXPAND_ENUM(threadIdx);
			auto th_idx = i * shp.n() + threadIdx.toid(shp);
			for (auto r : local_lds[th_idx]) {
				if (r.shfl) {
					tb_sim->ld(r.addr, r.sz, r.caller, r.shfl, r.scale, threadIdx);
				} else {
					tb_sim->ld(r.addr, r.sz, r.caller, threadIdx);
				}
			}
			for (auto r : local_shfls[th_idx]) {
				tb_sim->shfl(r.shfl, threadIdx);
			}
		}
	}

	return cusim->calculate();
}

