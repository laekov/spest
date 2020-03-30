#pragma once
#ifndef CU_SIM_H
#define CU_SIM_H

#include "tb_sim.h"

#include <vector>

struct CUSim {
	std::vector<TBSim*> tbs;
	void addTB(TBSim*);
	cnt_t calculate();
};

#endif
