#pragma once
#ifndef SIM_MEM_H
#define SIM_MEM_H

#include "shfl.h"
#include "tracer.h"
#include "hash.h"

template <class T>
class ROTensor {
private:
	Tracer* t;
	T* arr;

public:
	ROTensor(T* arr_, Tracer* t_=0): arr(arr_) , t(t_) {}

	T operator ()(const unsigned long addr, const char* file=__builtin_FILE(), 
			const int line=__builtin_LINE()) const {
		hash_t caller = hashCallerInfo(file, line);
		if (arr) {
			t->ld(arr + addr, caller);
			return arr[addr];
		} else {
			t->ld((T*)this + addr, caller);
			return 0;
		}
	}

	T operator ()(const ShflOut<int>& sout, const char* file=__builtin_FILE(), 
			const int line=__builtin_LINE()) const {
		return 0;
	}
};

#define SET_TRACER(_T_) Tracer* _default_tracer_ = &_T_
#define REG_RO_TENSOR(_T_, _V_) ROTensor<_T_> _V_##_(this->_V_, _default_tracer_)

#endif  // SIM_MEM_H
