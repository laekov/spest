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
	inline unsigned long toid(const TDim& sz) {
		return x + y * sz.x + z * sz.x * sz.y;
	}
};

#define ENUM_TDIM(iter, lim) \
	for (int iter##z = 0; iter##z < lim.z; ++iter##z) \
		for (int iter##y = 0; iter##y < lim.y; ++iter##y) \
			for (int iter##x = 0; iter##x < lim.x; ++iter##x) \

#define EXPAND_ENUM(iter) \
	TDim iter(iter##x, iter##y, iter##z)

#define SPEST_KERNEL_DEF_ARGS \
	TDim blockDim, TDim blockIdx, TDim threadIdx, Tracer* _tracer_

#define SPEST_KERNEL_ARGS \
	blockDim, blockIdx, threadIdx, _default_tracer_ 

#define SPEST_LAUNCH_KERNEL(__kernel__, n_blocks, n_threads) \
	ENUM_TDIM(blockIdx, n_blocks) ENUM_TDIM(threadIdx, n_threads) { \
		EXPAND_ENUM(blockIdx); \
		EXPAND_ENUM(threadIdx); \
		_default_tracer_->registerThread(blockIdx, threadIdx); \
		__kernel__; \
	}


#endif  // TDIM_H
