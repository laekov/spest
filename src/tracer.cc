#include <spest/tracer.h>


void Tracer::dims(TDim blocks, TDim threads) {
	sz = blocks;
	shp = threads;
}

void Tracer::setWFLimit(int n) {
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

void Tracer::ld(void* addr, size_t sz) {
	current_tb->ld(addr, sz);
}

cnt_t Tracer::get() const {
	if (current_tb) {
		cusim->addTB(current_tb);
	}
	return cusim->calculate();
}

