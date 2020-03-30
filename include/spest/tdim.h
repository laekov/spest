#pragma once
#ifndef TDIM_H
#define TDIM_H

struct TDim {
	unsigned long x, y, z;
	TDim(unsigned long x_=1, unsigned long y_=1, unsigned long z_=1):
		x(x_), y(y_), z(z_) {}
};

#define ENUM_TDIM(iter, lim) \
	for (TDim iter(0); iter.x < lim.x; ++iter.x) \
		for (iter.y = 0; iter.y < lim.y; ++iter.y) \
			for (iter.z = 0; iter.z < lim.z; ++iter.z)

#endif  // TDIM_H
