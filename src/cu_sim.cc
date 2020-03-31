#include <spest/cu_sim.h>
#include <spest/hw_spec.h>

void CUSim::addTB(TBSim* tb) {
	tbs.push_back(tb);
}

cnt_t CUSim::calculate() {
	cnt_t tot_trans = 0;
	auto gpu = HwSpec::getPlatform("vegavii");
	unsigned long num_th = gpu->max_threads;
	if (max_tb < 0x7ffff) {
		num_th = std::min(num_th, tbs[0]->sz.n() * max_tb);
	}
	for (auto tb : tbs) {
		tot_trans += tb->calculate(num_th);
	}
	return tot_trans;
}
