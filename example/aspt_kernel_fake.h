#include "spest/tensor.h"
#include "spest/tdim.h"

void aspt_dense_kernel_fake(int sc, 
		ROTensor<int> ptr, ROTensor<int> idx, 
		ROTensor<float> val, ROTensor<float> B, ROTensor<float> C,
		ROTensor<int> d_ptr, ROTensor<int> d_row, ROTensor<int> d_idx,
		ROTensor<int> r_ptr, ROTensor<int> r_idx,
		int Tr, int Tb, int Tc, TDim blockDim, TDim blockIdx, TDim threadIdx
		) {
	int db_id = blockIdx.x;
	int d_tile_id = blockIdx.y;

	// __shared__ float cB(Tc)(Tb);

	int d_ptr_begin = d_ptr(db_id), d_ptr_end = d_ptr(db_id + 1);

	/* load D into shared memory
	 * Tb threads as a load group(lg) to load a row in D
	 * there are in total WG_SIZE groups, so the stride of rows is WG_SIZE
	 */
	int lg_id = threadIdx.y;
	int n_lgs = blockDim.y;
	int lg_rk = threadIdx.x;
	int colb = lg_rk + d_tile_id * Tb;
	if (colb < sc) {
		for (int i = d_ptr_begin + lg_id; i < d_ptr_end; i += n_lgs) {
			B(d_idx(i) * sc + colb);
		}
	}
	// __syncthreads();

	/* Compute rows in S
	 * Tb threads as a compute group to compute a row in S and Tb cols in D
	 */
	int rpanel_id = d_row(db_id);
	int first_row = rpanel_id * Tr;
	if (colb < sc) {
		for (int row = first_row + lg_id; row < first_row + Tr; row += n_lgs) {
			int  row_end = ptr(row + 1);
			int r_idx_idx = r_ptr(rpanel_id) * Tr
				+ (row - first_row) * (r_ptr(rpanel_id + 1) - r_ptr(rpanel_id))
				+ db_id - (r_ptr(rpanel_id) - rpanel_id);
			int col = r_idx(r_idx_idx);
			int col_end = r_idx(r_idx_idx + 1);
			float prod = 0;
			for (; col < col_end; col += Tb) {
				int idx_local;
				float val_local;
				if (col + lg_rk >= col_end) {
					idx_local = -1;
				} else { 
					idx_local = idx(col + lg_rk);
					val_local = val(col + lg_rk);
				}
			// atomicAdd(C + row * sc + colb, prod);
			}
		}
	}
}

