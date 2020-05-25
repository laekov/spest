#pragma once
#ifndef TRACER_H
#define TRACER_H

#include <fstream>
#include <string>
#include <map>
#include <assert.h>

#include "tdim.h"
#include "tb_sim.h"
#include "cu_sim.h"
#include "hash.h"

struct Tracer {
	TDim sz, shp;

	CUSim *cusim;

	std::vector<TDim> block_idxs, thread_idxs;
	bool initialized;

	unsigned long getIdxByThread();

	void ld(void* addr, size_t sz, hash_t caller);
	void ld(void* addr, size_t sz, hash_t caller, class ShflOp* shfl, size_t scale);

	Tracer(): initialized(false) {}

	void sim(TDim blocks, TDim threads);

	void limitTBperCU(int n);

	void registerThread(int, TDim, TDim);

	void shfl(class ShflOp*);

	template<class T>
	void ld(T* addr, hash_t caller) {
		this->ld((void*)addr, sizeof(T), caller);
	}

	template<class T>
	void ld(T* addr, class ShflOp* shfl, size_t scale, hash_t caller) {
		this->ld((void*)addr, sizeof(T), caller, shfl, scale);
	}

	cnt_t get() const;
};


#endif  // TRACER_H
