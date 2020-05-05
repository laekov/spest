#pragma once
#ifndef TDIM_H
#define TDIM_H

typedef double cnt_t;

struct TDim {
	unsigned long x, y, z;
	TDim(unsigned long x_=1, unsigned long y_=1, unsigned long z_=1):
		x(x_), y(y_), z(z_) {}
	inline unsigned long n() const {
		return x * y * z;
	}
};

#define ENUM_TDIM(iter, lim) \
	for (TDim iter(0, 0, 0); iter.z < lim.z; ++iter.z) \
		for (iter.y = 0; iter.y < lim.y; ++iter.y) \
			for (iter.x = 0; iter.x < lim.x; ++iter.x) \

#define SPEST_KERNEL_DEF_ARGS \
	TDim blockDim, TDim blockIdx, TDim threadIdx, Tracer* _tracer_

#define SPEST_KERNEL_ARGS \
	blockDim, blockIdx, threadIdx, _default_tracer_ 

#endif  // TDIM_H
