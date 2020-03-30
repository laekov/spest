#pragma once
#ifndef SIM_MEM_H
#define SIM_MEM_H

#include "tracer.h"

template <class T>
class ROTensor {
private:
	Tracer* t;
	T* arr;

public:
	ROTensor(T* arr_, Tracer* t_=0): arr(arr_) , t(t_) {}

	T operator [](const unsigned long addr) const {
		if (arr) {
			t->ld(arr + addr);
			return arr[addr];
		} else {
			t->ld((T*)this + addr);
			return 0;
		}
	}
};

#define SET_TRACER(_T_) Tracer* _default_tracer_ = &_T_
#define REG_RO_TENSOR(_T_, _V_) ROTensor<_T_> _V_##_(this->_V_, _default_tracer_)

#endif  // SIM_MEM_H
