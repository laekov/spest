#pragma once
#ifndef HW_SPEC
#define HW_SPEC

#include <unordered_map>
#include <string>

typedef std::unordered_map<unsigned long long, double> prof_res_t;

struct HwSpec {
	static inline unsigned long long encodeKey(int a, int b, int c) {
		return ((unsigned long long)a << 40) | ((unsigned long long)b << 20) | c;
	}
	static std::unordered_map<std::string, HwSpec*> platforms;
	static HwSpec* getPlatform(std::string);


	int max_threads;
	prof_res_t global_mem_lat;
	double mean_global_mem_lat;

	double getGlobalMemBw(int threads, int width, int same);
};

#endif  // HW_SPEC
