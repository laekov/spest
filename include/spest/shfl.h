#pragma once
#ifndef SHFL_H
#define SHFL_H

#include <spest/tdim.h>
#include <spest/tracer.h>

template<class T>
class ShflOut {
public:
	T val;
	int tgt_rank, gran;
	int offset, scale;

	ShflOut(T v_, int t_, int g_, int o_=0, int s_=1):
		val(v_), tgt_rank(t_), gran(g_), offset(o_), scale(s_) {}

	ShflOut<T> operator +(const int& o) {
		return ShflOut(val, tgt_rank, gran, offset + o, scale);
	}
};


template<class T>
ShflOut<T> createShuffle(Tracer* t, T val, int tgt, int gran, TDim idx);

#define __shfl(_v_, _t_, _g_) createShuffle(_tracer_, _v_, _t_, _g_, threadIdx)

#endif  // SHFL_H
