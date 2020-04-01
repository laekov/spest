#include <set>
#include <map>
#include <vector>
#include <queue>
#include <functional>

#include "spbenchmark/common/test.h"

#include "spest/tracer.h"
#include "spest/tensor.h"

#include "aspt_kernel_fake.h"

#define SP_FLAG (1<<30)
#define CEIL(a, b) ((a) == 0 ? 0 : ((a - 1) / (b) + 1))


void* hAllocAndCopy(void* a, int sz) {
	void* b = new char[sz];
	memcpy(b, a, sz);
	return b;
}

class sDenseMemSim: public Tester<float> {
public:
	int WG_SIZE, Tr, Tc, Tb;

protected:

	/* nr0: original number of rows in A
	 * nr: number of rows, aligned
	 * sc: number of columns in B and C
	 * w: number of rows in B
	 * nnz: number of non-zeros in A
	 */
	int nr0, nr, sc, w, nnz;

	/*
	 * nb: amount of row panels
	 * n_db: amount of dense blocks
	 */
	int nb, n_db;

	/*
	 * d_idx: column indices of dense blocks
	 * d_ptr: n_db + 1 entries, indicating start index of the each dense block in d_idx
	 * d_row: n_db entries, indicating dense block's row panel number
	 */
	int *d_ptr, *d_idx, *d_row;
	int *_d_ptr, *_d_idx, *_d_row;

	/*
	 * sp_ptr: pointer to first sparse tile element in each row
	 */
	int *sp_ptr, *_sp_ptr;

	/*
	 * r_idx: dense tile start index of columns
	 * r_d_ptr: nb + 1 entries, indicating start position of denseblo
	 */
	int *r_idx, *r_ptr;
	int *_r_idx, *_r_ptr;

	int *ptr, *idx;
	float *val;

	int *_ptr, *_idx;
	float *_val;
	float *_B, *vin;
	float *_vout, *vout;


private:
	void figureOutDenseBlocks() {
		std::vector<int> dense_idxs, d_row_v, d_ptr_v;
		sp_ptr = new int[nr];

		r_ptr = new int[nb + 1];
		std::vector<int> r_idx_v;

		n_db = 0;
		for (int i = 0; i < nb; ++i) {
			std::map<int, int> count;
			std::set<int> denses;
			
			// Find dense columns
			for (int j = ptr[i * Tr]; j < ptr[(i + 1) * Tr]; ++j) {
				denses.insert(idx[j]);
			}

			// Add dense column indices into the vector
			for (int j = 0; j < denses.size(); j += Tc) {
				d_row_v.push_back(i);
				d_ptr_v.push_back(dense_idxs.size() + j);
			}
			int n_db_row = CEIL(denses.size(), Tc);
			n_db += n_db_row;
			int idx_base = dense_idxs.size();

			// TODO : do column sort here

			for (const int& j : denses) {
				dense_idxs.push_back(j);
			}

			// Mark dense columns and sort them in each row
			for (int j = i * Tr; j < (i + 1) * Tr; ++j) {
				std::vector<std::pair<int, float> > entries;
				int u, k;
				for (u = ptr[j]; u < ptr[j + 1]; ++u) {
					if (denses.find(idx[u]) == denses.end()) {
						idx[u] |= SP_FLAG;
					}
					entries.push_back(std::pair<int, float>(idx[u], val[u]));
				}
				// TODO : sort according to dense column arrangement
				std::sort(entries.begin(), entries.end());

				/* put entries back
				 * for dense columns, use the index in d_idx instead
				 */
				for (u = ptr[j], k = idx_base; u < ptr[j + 1] 
						&& !(entries[u - ptr[j]].first & SP_FLAG); ++u) {
					int eidx = entries[u - ptr[j]].first;
					for (; eidx != dense_idxs[k]; ++k);
					idx[u] = k;
					val[u] = entries[u - ptr[j]].second;
				}
				int dense_end = u;
				sp_ptr[j] = u;
				for (; u < ptr[j + 1]; ++u) {
					idx[u] = entries[u - ptr[j]].first ^ SP_FLAG;
					val[u] = entries[u - ptr[j]].second;
				}

				/* calculate r_idx by iterate through the row */
				for (k = 0, u = ptr[j]; k < n_db_row; ++k) {
					r_idx_v.push_back(u);
					for (; u < dense_end && (idx[u] - idx_base) < (k + 1) * Tc; ++u);
				}
				r_idx_v.push_back(dense_end);
			}
			r_ptr[i + 1] = n_db_row + 1;
		}

		d_row = new int[n_db];
		std::copy(d_row_v.begin(), d_row_v.end(), d_row);

		d_ptr = new int[n_db + 1];
		d_ptr_v.push_back(dense_idxs.size());
		std::copy(d_ptr_v.begin(), d_ptr_v.end(), d_ptr);

		d_idx = new int[d_ptr[n_db]];
		std::copy(dense_idxs.begin(), dense_idxs.end(), d_idx);

		r_ptr[0] = 0;
		for (int i = 1; i <= nb; ++i) {
			r_ptr[i] += r_ptr[i - 1];
		}
		int sz_ridx = r_idx_v.size();
		r_idx = new int[sz_ridx];
		std::copy(r_idx_v.begin(), r_idx_v.end(), r_idx);
	}

public:
	void data2Device(CSRMatrix<float> A, float* B, int k, int m) {

		nr0 = nr = A.n;
		nnz = A.nnz;
		sc = m;
		w = k;
		nb = CEIL(nr, Tr);
		nr = nb * Tr;

		ptr = new int[nr + 1];
		memcpy(ptr, A.ptr, (nr0 + 1) * sizeof(int));
		for (int i = nr0 + 1; i <= nr; ++i) {
			ptr[i] = ptr[nr0];
		}

		idx = (int*)hAllocAndCopy(A.idx, nnz * sizeof(int));
		val = (float*)hAllocAndCopy(A.val, nnz * sizeof(float));

		figureOutDenseBlocks();

		vin = new float[w * sc];
		for (int i = 0; i < w; ++i) {
			for (int j = 0; j < sc; ++j) {
				vin[i * sc + j] = B[i + j * w];
			}
		}

		vout = new float[nr * sc];
	}

	void clean() {

		delete [] vin;

		delete [] d_ptr;
		delete [] d_idx;
		delete [] d_row;

		delete [] ptr;
		delete [] idx;
		delete [] val;

		delete [] vout;
	}

	float* device2Host() {
		return vout;
	}

	sDenseMemSim(int _0, int _1, int _2, int _3):
		WG_SIZE(_0), Tr(_1), Tb(_2), Tc(_3) {}

	virtual void compute() {
		TDim blockn(n_db, CEIL(sc, Tb));
		TDim blockDim(Tb, WG_SIZE);

		Tracer t;

		t.dims(blockn, blockDim);
		t.limitTBperCU(16384 / (Tb * Tc));

		SET_TRACER(t);
		REG_RO_TENSOR(int, ptr);
		REG_RO_TENSOR(int, idx);
		REG_RO_TENSOR(float, val);
		REG_RO_TENSOR(float, vin);
		REG_RO_TENSOR(float, vout);
		REG_RO_TENSOR(int, d_ptr);
		REG_RO_TENSOR(int, d_row);
		REG_RO_TENSOR(int, d_idx);
		REG_RO_TENSOR(int, r_ptr);
		REG_RO_TENSOR(int, r_idx);
		/*
		ROTensor<int> ptr_(ptr, t), idx_(idx, t);
		ROTensor<float> val_(val, t), B_(vin, t), C_(vout, t);
		ROTensor<int> d_ptr_(d_ptr, t), d_row_(d_row, t), d_idx_(d_idx, t);
		ROTensor<int> r_ptr_(r_ptr, t), r_idx_(r_idx, t);
		*/

		// SPOUT(blockn.x << " " << blockn.y);
		// SPOUT(blockDim.x << " " << blockDim.y);

		ENUM_TDIM(blockIdx, blockn) {
			t.block(blockIdx);
			ENUM_TDIM(threadIdx, blockDim) {
				t.thread(threadIdx);
				aspt_dense_kernel_fake(sc,
					ptr_, idx_,
					val_, vin_, vout_,
					d_ptr_, d_row_, d_idx_,
					r_ptr_, r_idx_,
					Tr, Tb, Tc, blockDim, blockIdx, threadIdx);
			}
		}
		SPOUT(t.get());
	}
};


int main(int argc, char* args[]) {
	if (argc != 6) {
		SPLOG("Wrong number of arguments");
		return 1;
	}

	int tile_params[4];
	for (int i = 0; i < 4; ++i) {
		tile_params[i] = atoi(args[i + 2]);
	}
	Tester<float>* tester = new sDenseMemSim(tile_params[0], tile_params[1], 
			tile_params[2], tile_params[3]);

	std::string filename(args[1]);
	CSRMatrix<float> A;
	float *B;
	int m = 32;
	int n, k;
	prepareData(filename, A, B, n, k, m);
	tester->data2Device(A, B, k, m);
	tester->compute();
}
