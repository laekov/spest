#pragma once
#ifndef MEM_ACCESS_H
#define MEM_ACCESS_H

struct MemAccess {
	size_t sz;
	unsigned long addr;

	MemAccess(size_t sz_, void* addr_): sz(sz_), addr((unsigned long)addr_) {}
};

inline bool operator <(const MemAccess& a, const MemAccess& b) {
	return a.addr < b.addr;
}

#endif
