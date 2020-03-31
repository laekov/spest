#pragma once
#ifndef CU_SIM_H
#define CU_SIM_H

#include "tb_sim.h"

#include <vector>

struct CUSim {
	int max_th, max_tb;
	std::vector<TBSim*> tbs;
	void addTB(TBSim*);
	cnt_t calculate();

	CUSim(): max_th(0x7ffff), max_tb(0x7ffff) {}
};

#endif
