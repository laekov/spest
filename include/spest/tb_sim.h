#pragma once
#ifndef TB_SIM_H
#define TB_SIM_H

#include <vector>
#include "tdim.h"
#include "mem_access.h"

struct TBSim {
	TDim sz;
	std::vector<std::vector<MemAccess> > global_lds;
	std::vector<std::vector<MemAccess> > shared_lds;

	int current_thread;

	TBSim(TDim dims): sz(dims), current_thread(0) {
		global_lds.resize(dims.n());
	}

	void ld(void*, size_t);
	void nextThread();
	cnt_t calculate();
};

#endif  // TB_SIM_H
