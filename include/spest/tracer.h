#pragma once
#ifndef TRACER_H
#define TRACER_H

#include <fstream>
#include <string>
#include <assert.h>

#include "tdim.h"
#include "tb_sim.h"
#include "cu_sim.h"


struct Tracer {
	TDim sz, shp;
	TBSim *current_tb;
	CUSim *cusim;

	void ld(void* addr, size_t sz);

	Tracer(): current_tb(0), cusim(new CUSim) {}

	void dims(TDim blocks, TDim threads);

	void setWFLimit(int n);

	void block(TDim idx);

	void thread(TDim idx);

	template<class T>
	void ld(T* addr) {
		this->ld((void*)addr, sizeof(T));
	}

	cnt_t get() const;
};

#endif  // TRACER_H
