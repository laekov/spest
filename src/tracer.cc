#include <spest/tracer.h>


void Tracer::dims(TDim blocks, TDim threads) {
	sz = blocks;
	shp = threads;
}

void Tracer::limitTBperCU(int n) {
	cusim->max_tb = std::min(cusim->max_tb, n);
}

void Tracer::block(TDim idx) {
	if (current_tb) {
		cusim->addTB(current_tb);
	}
	current_tb = 0;
}

void Tracer::thread(TDim idx) {
	if (current_tb) {
		current_tb->nextThread();
	} else {
		current_tb = new TBSim(shp);
	}
}

void Tracer::ld(void* addr, size_t sz, hash_t caller) {
	current_tb->ld(addr, sz, caller);
}

void Tracer::ld(void* addr, size_t sz, hash_t caller, class ShflOp* shfl, size_t scale) {
	current_tb->ld(addr, sz, caller, shfl, scale);
}

void Tracer::shfl(ShflOp* op) {
	current_tb->shfl(op);
}

cnt_t Tracer::get() const {
	if (current_tb) {
		cusim->addTB(current_tb);
	}
	return cusim->calculate();
}

