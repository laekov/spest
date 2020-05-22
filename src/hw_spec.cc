#include <unistd.h>
#include <assert.h>
#include <fstream>
#include <algorithm>

#include <spest/hw_spec.h>
#include <spest/debug.h>

std::unordered_map<std::string, HwSpec*> HwSpec::platforms;

#define RECOGNIZE_PARAM(__key__) \
	if (key == #__key__) { \
		h->__key__ = atoi(value.c_str()); \
	}

HwSpec* HwSpec::getPlatform(std::string platform) {
	if (platform == "system") {
		platform = std::string(std::getenv("SPEST_PLATFORM"));
	}
	if (platforms.find(platform) != platforms.end()) {
		return platforms[platform];
	}

	HwSpec* h = new HwSpec;
	{
		auto spec_file = std::string(std::getenv("HOME")) + "/.spest/"
			+ platform + ".ini";
		std::ifstream fin(spec_file);
		if (!fin.is_open()) {
			SPEST_LOG("Spec file " + spec_file + " does not exist");
			assert(fin.is_open());
		}
		std::string line;
		while (fin >> line) {
			auto p0 = line.find('=');
			auto key = line.substr(0, p0);
			auto value = line.substr(p0 + 1);
			RECOGNIZE_PARAM(max_threads)
			RECOGNIZE_PARAM(max_tb_per_cu)
			RECOGNIZE_PARAM(num_cu)
		}
		fin.close();
	}

	{
		auto perf_file = std::string(std::getenv("HOME")) + "/.spest/" 
			+ platform + ".perf.in";
		std::ifstream fin(perf_file);
		if (!fin.is_open()) {
			SPEST_LOG("Performance record file " + perf_file + " does not exist");
			assert(fin.is_open());
		}
		int same, width, threads;
		double gflops;
		int n = 0;
		prof_res_t cnt_threads;
		while (fin >> same >> width >> threads >> gflops) {
			h->global_mem_lat[encodeKey(same, width, threads)] = 1. / gflops;
			h->mean_global_mem_lat[threads] += 1. / gflops;
			cnt_threads[threads] += 1.;
			++n;
		}
		fin.close();

		for (auto& i : h->mean_global_mem_lat) {
			i.second /= cnt_threads[i.first];
		}
	}

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
	SPEST_LOG("Cannot find bandwidth for " << threads << " " << same << " " 
			<< width << " using " << sm << " " << wr << "\n");
	return mean_global_mem_lat[threads];
}
