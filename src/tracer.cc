#include <omp.h>

#include <spest/tracer.h>


void Tracer::sim(TDim blocks, TDim threads) {
	sz = blocks;
	shp = threads;
	cusim = new CUSim(blocks, threads);
}

void Tracer::limitTBperCU(int n) {
	cusim->max_tb = std::min(cusim->max_tb, n);
}

void Tracer::registerThread(int omp_thread_idx, TDim blockIdx, TDim threadIdx) {
#pragma omp critical
	block_idxs[omp_thread_idx] = blockIdx;
#pragma omp critical
	thread_idxs[omp_thread_idx] = threadIdx;
}

void Tracer::ld(void* addr, size_t sz, hash_t caller) {
	getTBSim()->ld(addr, sz, caller, getTh());
}

void Tracer::ld(void* addr, size_t sz, hash_t caller, class ShflOp* shfl, size_t scale) {
	getTBSim()->ld(addr, sz, caller, shfl, scale, getTh());
}

void Tracer::shfl(ShflOp* op) {
	getTBSim()->shfl(op, getTh());
}

TBSim* Tracer::getTBSim() {
	return cusim->tbs[block_idxs[omp_get_thread_num()].toid(shp)];
}

TDim Tracer::getTh() {
	return thread_idxs[omp_get_thread_num()];
}

cnt_t Tracer::get() const {
	return cusim->calculate();
}

