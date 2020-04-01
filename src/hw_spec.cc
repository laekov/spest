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
	int width, threads;
	double gflops;
	double tot = 0;
	int n = 0;
	HwSpec* h = new HwSpec;
	fin >> h->max_threads;
	while (fin >> width >> threads >> gflops) {
		h->global_mem_lat[encodeKey(width, threads)] = 1e9 / gflops;
		tot += 1e9 / gflops;
		++n;
	}
	h->mean_global_mem_lat = tot / std::max(n, 1);
	fin.close();
	platforms[platform] = h;
	return h;
}

double HwSpec::getGlobalMemLat(int threads, int width) {
	auto key = encodeKey(width, threads);
	auto it = global_mem_lat.find(key);
	if (it != global_mem_lat.end()) {
		return it->second;
	}
	int wr = 1;
	for (; wr < width; wr <<= 1);
	key = encodeKey(wr, threads);
	it = global_mem_lat.find(key);
	if (it != global_mem_lat.end()) {
		return it->second;
	}
	// SPEST_LOG(width);
	return mean_global_mem_lat;
}
