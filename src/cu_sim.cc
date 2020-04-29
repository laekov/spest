#include <spest/cu_sim.h>
#include <spest/hw_spec.h>

#include <queue>
#include <vector>
#include <functional>

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
	std::priority_queue<cnt_t, std::vector<cnt_t>, std::greater<cnt_t> > cu_time;
	for (int i = 0; i < 60; ++i) {
		cu_time.push(0);
	}
	for (auto tb : tbs) {
		auto t = tb->calculate(num_th);
		tot_trans += t;
		t += cu_time.top();
		cu_time.pop();
		cu_time.push(t);
	}
	cnt_t t;
	while (!cu_time.empty()) {
		t = cu_time.top();
		cu_time.pop();
	}
	return t;
}
