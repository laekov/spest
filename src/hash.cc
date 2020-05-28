#include <spest/hash.h>
#include <spest/debug.h>

#include <sstream>
#include <unordered_map>

std::unordered_map<hash_t, std::pair<std::string, int> > _hash_record;

std::string lookupHashRecord(hash_t hash) {
	auto it = _hash_record.find(hash);
	if (it == _hash_record.end()) {
		return "Hash not found";
	}
	std::ostringstream oss;
	oss << it->second.first << ":" << it->second.second;
	return oss.str();
}

hash_t hashCallerInfo(const char* file, int line) {
	hash_t res = 0;
	for (const char* i = file; *i; ++i) {
		res = res * 37 + *i;
	}
#ifdef SPEST_DEBUG
	if (_hash_record.find(res ^ line) == _hash_record.end()) {
#pragma omp critical
		_hash_record[res ^ line] = std::pair<std::string, int>(std::string(file), line);
	}
#endif
	return res ^ line;
}
