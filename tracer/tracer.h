#pragma once
#ifndef TRACER_H
#define TRACER_H

#include "tdim.h"

#include <fstream>
#include <string>
#include <assert.h>

typedef unsigned long long memcnt_t;

class Tracer {
private:
	memcnt_t n;
	std::ofstream fou;

public:
	Tracer(std::string filename) : n(0), fou(filename) {}

	void dims(TDim blocks, TDim threads) {
		fou << "META.LAUNCH_DIMS " << blocks.x << " " << blocks.y << " " << blocks.z << " "
			<< threads.x << " " << threads.y << " " << threads.z << std::endl;
	}

	void setWFLimit(int n) {
		fou << "META.WF_LIMIT " << n << std::endl;
	}

	void block(TDim idx) {
		fou << "LAUNCH_THREAD " << idx.x << " " << idx.y << " " << idx.z << std::endl;
	}

	void thread(TDim idx) {
		fou << "LAUNCH_THREAD " << idx.x << " " << idx.y << " " << idx.z << std::endl;
	}

	template<class T>
	void ld(T* addr) {
		assert(fou.is_open());
		fou << "LD.GLOBAL " << sizeof(T) << " " << (unsigned long)addr << std::endl;
		n += sizeof(T);
	}

	template<class T>
	void atomic(T* addr) {
		fou << "ATOMIC.GLOBAL " << sizeof(T) << " " << (unsigned long)addr << std::endl;
		// n += sizeof(T);
	}

	memcnt_t get() const {
		return n;
	}

	// TODO
	// void parallelAccess() {}
};

#endif  // TRACER_H
