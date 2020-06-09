#pragma once
#ifndef CU_SIM_H
#define CU_SIM_H

#include "tb_sim.h"

#include <vector>

struct CUSim {
	int max_th, max_tb;
	std::vector<TBSim*> tbs;
	std::vector<cnt_t> tb_res;
	struct HwSpec* gpu;
	TDim sz, shp;

	cnt_t calculate();

	struct HwSpec* getGPU();

	unsigned long getNumTh();

	CUSim(TDim sz_, TDim shp_): max_th(0x7ffff), max_tb(0x7ffff), gpu(0) {
		sz = sz_;
		shp = shp_;
		tbs.resize(sz_.n());
		for (auto& tb : tbs) {
			tb = new TBSim(shp_);
		}
		tb_res.resize(tbs.size());
	}
};

#endif
