#include <omp.h>
#include <vector>

#include <spest/shfl.h>


const int max_th = 1024;
const int chunk_size = 1 << 10;;

ShflOp *shfl_ptr[max_th] = {0}, *shfl_end[max_th] = {0};

std::vector<ShflOp*> chunks;

ShflOp* allocShfl(int a, int b, size_t c) {
	int thid = omp_get_thread_num();
	if (shfl_ptr[thid] == shfl_end[thid]) {
		shfl_end[thid] = (shfl_ptr[thid] = new ShflOp[chunk_size]) + chunk_size;
#pragma omp critical
		chunks.push_back(shfl_ptr[thid]);
	}
	ShflOp* op = shfl_ptr[thid]++;
	op->tgt_rank = a;
	op->gran = b;
	op->sz = c;
	return op;
}

void clearShfl() {
	for (auto i : chunks) {
		delete [] i;
	}
	chunks.clear();
}
