#include <spest/mem_access.h>
#include <spest/shfl.h>

#include <cstring>

unsigned long MemAccess::getAddr() {
	if (!shfl) {
		return addr;
	}
	int offset;
	memcpy(&offset, shfl->res, shfl->sz);
	return addr + offset * scale;
}
