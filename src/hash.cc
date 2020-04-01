#include <spest/hash.h>

hash_t hashCallerInfo(const char* file, int line) {
	hash_t res = 0;
	for (const char* i = file; *i; ++i) {
		res = res * 37 + *i;
	}
	return res ^ line;
}
