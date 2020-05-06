#pragma once
#ifndef MEM_ACCESS_H
#define MEM_ACCESS_H

#include <cstddef>

struct MemAccess {
	size_t sz;
	unsigned long addr;
	size_t scale;
	class ShflOp* shfl;

	MemAccess(size_t sz_, void* addr_, class ShflOp* shfl_=0): 
		sz(sz_), addr((unsigned long)addr_), shfl(shfl_) {}

	unsigned long getAddr();
};

inline bool operator <(const MemAccess& a, const MemAccess& b) {
	return a.addr < b.addr;
}

#endif
