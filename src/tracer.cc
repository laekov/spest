#include <spest/tracer.h>


void Tracer::dims(TDim blocks, TDim threads) {
	fou << "META.LAUNCH_DIMS " << blocks.x << " " << blocks.y << " " << blocks.z << " "
		<< threads.x << " " << threads.y << " " << threads.z << std::endl;
}

void Tracer::setWFLimit(int n) {
	fou << "META.WF_LIMIT " << n << std::endl;
}

void Tracer::block(TDim idx) {
	fou << "LAUNCH_THREAD " << idx.x << " " << idx.y << " " << idx.z << std::endl;
}

void Tracer::thread(TDim idx) {
	fou << "LAUNCH_THREAD " << idx.x << " " << idx.y << " " << idx.z << std::endl;
}

void Tracer::ld(void* addr, size_t sz) {
	fou << "ATOMIC.GLOBAL " << sz << " " << (unsigned long)addr << std::endl;
	n += sz;
}

cnt_t Tracer::get() const {
	return n;
}

