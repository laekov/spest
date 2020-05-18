#include <unistd.h>
#include <assert.h>
#include <fstream>
#include <algorithm>

#include <spest/hw_spec.h>
#include <spest/debug.h>

std::unordered_map<std::string, HwSpec*> HwSpec::platforms;

HwSpec* HwSpec::getPlatform(std::string platform) {
	if (platforms.find(platform) != platforms.end()) {
		return platforms[platform];
	}
	auto spec_file = std::string(std::getenv("HOME")) + "/.spest/" 
		+ platform + ".spec";
	std::ifstream fin(spec_file);
	if (!fin.is_open()) {
		SPEST_LOG("Spec file " + spec_file + " does not exist");
		assert(fin.is_open());
	}
	int same, width, threads;
	double gflops;
	double tot = 0;
	int n = 0;
	HwSpec* h = new HwSpec;
	fin >> h->max_threads;
	while (fin >> same >> width >> threads >> gflops) {
		h->global_mem_lat[encodeKey(same, width, threads)] = 1. / gflops;
		tot += 1. / gflops;
		++n;
	}
	h->mean_global_mem_lat = tot / std::max(n, 1);
	fin.close();
	platforms[platform] = h;
	return h;
}

double HwSpec::getGlobalMemBw(int threads, int width, int same) {
	auto key = encodeKey(same, width, threads);
	auto it = global_mem_lat.find(key);
	if (it != global_mem_lat.end()) {
		return it->second;
	}
	int sm = 1;
	for (; (sm << 1) <= same; sm <<= 1);
	int wr = 4;
	for (; wr < width; wr <<= 1);
	key = encodeKey(sm, wr, threads);
	it = global_mem_lat.find(key);
	if (it != global_mem_lat.end()) {
		return it->second;
	}
	SPEST_LOG("Cannot find bandwidth for " << threads << " " << same << " " << width) << " using " << sm << " " << wr << "\n";
	exit(0);
	return mean_global_mem_lat;
}
