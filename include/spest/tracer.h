#pragma once
#ifndef TRACER_H
#define TRACER_H

#include "tdim.h"

#include <fstream>
#include <string>
#include <assert.h>

typedef unsigned long long cnt_t;

class Tracer {
private:
	cnt_t n;
	std::ofstream fou;

	void ld(void* addr, size_t sz);

public:
	Tracer(std::string filename) : n(0), fou(filename) {}

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
