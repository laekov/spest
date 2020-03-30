#include <spest/cu_sim.h>

void CUSim::addTB(TBSim* tb) {
	tbs.push_back(tb);
}

cnt_t CUSim::calculate() {
	cnt_t tot_trans = 0;
	for (auto tb : tbs) {
		tot_trans += tb->calculate();
	}
	return tot_trans;
}
