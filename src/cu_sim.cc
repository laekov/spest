#include <spest/cu_sim.h>
#include <spest/hw_spec.h>
#include <spest/debug.h>
#include <spest/shfl.h>

#include <queue>
#include <vector>
#include <functional>

cnt_t CUSim::calculate() {
	if (tbs.size() == 0) {
		return 0;
	}
	cnt_t tot_trans = 0;
	auto gpu = HwSpec::getPlatform("system");
	unsigned long num_th = gpu->max_threads;
	max_tb = std::min(max_tb, gpu->max_tb_per_cu);
	if (max_tb < 0x7ffff) {
		num_th = std::min(num_th, tbs[0]->sz.n() * max_tb);
	}
	std::priority_queue<cnt_t, std::vector<cnt_t>, std::greater<cnt_t> > cu_time;

	for (int i = 0; i < gpu->num_cu; ++i) {
		cu_time.push(0);
	}
	std::vector<cnt_t> tb_res;
	tb_res.resize(tbs.size());
	size_t n_tbs = tbs.size();
	// tbs[2]->calculate(num_th);
	// return 0;
#pragma omp parallel for schedule(dynamic, 4)
	for (size_t i = 0; i < n_tbs; ++i) {
		tb_res[i] = tbs[i]->calculate(num_th);
		delete tbs[i];
	}
	clearShfl();
	for (size_t i = 0; i < tbs.size(); ++i) {
		auto t = tb_res[i]; // tb->calculate(num_th);
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
