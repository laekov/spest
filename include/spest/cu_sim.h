#pragma once
#ifndef CU_SIM_H
#define CU_SIM_H

#include "tb_sim.h"

#include <vector>

struct CUSim {
	int max_th, max_tb;
	std::vector<TBSim*> tbs;

	cnt_t calculate();

	CUSim(TDim sz_, TDim shp_): max_th(0x7ffff), max_tb(0x7ffff) {
		tbs.resize(sz_.n());
		for (auto& tb : tbs) {
			tb = new TBSim(shp_);
		}
	}
};

#endif
