#pragma once
#ifndef TB_SIM_H
#define TB_SIM_H

#include <vector>
#include <unordered_map>

#include "tdim.h"
#include "hash.h"
#include "mem_access.h"

struct TBSim {
	TDim sz;
	std::unordered_map<hash_t, std::vector<std::vector<MemAccess> > > global_lds;

	std::vector<std::vector<class ShflOp*> > shfls;
	std::vector<std::vector<MemAccess> > shared_lds;

	int current_thread;

	TBSim(TDim dims): sz(dims), shfls(dims.n()) {}

	void checkCaller(hash_t);

	void ld(void*, size_t, hash_t, int);
	void ld(void*, size_t, hash_t, class ShflOp*, size_t, int);
	void shfl(class ShflOp*, int);
	void nextThread();

	int getTh(TDim idx);

	int resolveShfls();
	cnt_t calculate(int);
};

#endif  // TB_SIM_H
