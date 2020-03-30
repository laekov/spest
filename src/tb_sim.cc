#include <spest/tb_sim.h>

void TBSim::ld(void* addr, size_t sz) {
	global_lds[current_thread].push_back(MemAccess(sz, addr));
}

void TBSim::nextThread() {
	++current_thread;
}

cnt_t TBSim::calculate() {
	return 0;
}
