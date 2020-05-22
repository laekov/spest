#pragma once
#ifndef SHFL_H
#define SHFL_H

#include <spest/tdim.h>
#include <spest/tracer.h>

class ShflOp {
public:
	char val[8], res[8];
	int tgt_rank, gran;
	size_t sz;

	ShflOp(int t_, int g_, size_t sz_):
		tgt_rank(t_), gran(g_), sz(sz_) {}
};

template<class T>
class ShflOut {
public:
	ShflOp* op;
	T offset, scale;

	ShflOut(ShflOp* op_, int o_=0, int s_=1):
		op(op_), offset(o_), scale(s_) {}

	ShflOut<T> operator +(const T& o) {
		return ShflOut(op, offset + o, scale);
	}
};

template<class T>
ShflOut<T> fakeShuffle(Tracer* t, T val, int tgt, int gran, TDim idx) {
	ShflOp* op;
	op = new ShflOp(tgt, gran, sizeof(T));
	memcpy(op->val, &val, sizeof(T));

	t->shfl(op);
	return ShflOut<T>(op);
}

#define __shfl(_v_, _t_, _g_) fakeShuffle(_tracer_, _v_, _t_, _g_, threadIdx)

#endif  // SHFL_H
