#include "updates.h"
#include <tgmath.h>
#include "linalg.h"
#include "rand.h"
#include "util.h"
#include <stdio.h>

void update_delayed(const int N, const int n_delay, const double *const restrict del,
		const int *const restrict site_order,
		uint64_t *const restrict rng, int *const restrict hs,
		num *const restrict gu, num *const restrict gd, num *const restrict phase,
		num *const restrict au, num *const restrict bu, num *const restrict du,
		num *const restrict ad, num *const restrict bd, num *const restrict dd)
{
	int k = 0;
	for (int j = 0; j < N; j++) du[j] = gu[j + N*j];
	for (int j = 0; j < N; j++) dd[j] = gd[j + N*j];
	for (int ii = 0; ii < N; ii++) {
		const int i = site_order[ii];
		const double delu = del[i + N*hs[i]];
		const double deld = del[i + N*!hs[i]];
		if (delu == 0.0 && deld == 0.0) continue;
		const num ru = 1.0 + (1.0 - du[i]) * delu;
		const num rd = 1.0 + (1.0 - dd[i]) * deld;
		const num prob = ru * rd;
		const double absprob = fabs(prob);
		if (rand_doub(rng) < absprob) {
			#pragma omp parallel sections
			{
			#pragma omp section
			{
			for (int j = 0; j < N; j++) au[j + N*k] = gu[j + N*i];
			for (int j = 0; j < N; j++) bu[j + N*k] = gu[i + N*j];
			xgemv("N", N, k, 1.0, au, N, bu + i,
			      N, 1.0, au + N*k, 1);
			xgemv("N", N, k, 1.0, bu, N, au + i,
			      N, 1.0, bu + N*k, 1);
			au[i + N*k] -= 1.0;
			for (int j = 0; j < N; j++) au[j + N*k] *= delu/ru;
			for (int j = 0; j < N; j++) du[j] += au[j + N*k] * bu[j + N*k];
			}
			#pragma omp section
			{
			for (int j = 0; j < N; j++) ad[j + N*k] = gd[j + N*i];
			for (int j = 0; j < N; j++) bd[j + N*k] = gd[i + N*j];
			xgemv("N", N, k, 1.0, ad, N, bd + i,
			      N, 1.0, ad + N*k, 1);
			xgemv("N", N, k, 1.0, bd, N, ad + i,
			      N, 1.0, bd + N*k, 1);
			ad[i + N*k] -= 1.0;
			for (int j = 0; j < N; j++) ad[j + N*k] *= deld/rd;
			for (int j = 0; j < N; j++) dd[j] += ad[j + N*k] * bd[j + N*k];
			}
			}
			k++;
			hs[i] = !hs[i];
			*phase *= prob/absprob;
		}
		if (k == n_delay) {
			k = 0;
			#pragma omp parallel sections
			{
			#pragma omp section
			{
			xgemm("N", "T", N, N, n_delay, 1.0,
			      au, N, bu, N, 1.0, gu, N);
			for (int j = 0; j < N; j++) du[j] = gu[j + N*j];
			}
			#pragma omp section
			{
			xgemm("N", "T", N, N, n_delay, 1.0,
			      ad, N, bd, N, 1.0, gd, N);
			for (int j = 0; j < N; j++) dd[j] = gd[j + N*j];
			}
			}
		}
	}
	#pragma omp parallel sections
	{
	#pragma omp section
	xgemm("N", "T", N, N, k, 1.0, au, N, bu, N, 1.0, gu, N);
	#pragma omp section
	xgemm("N", "T", N, N, k, 1.0, ad, N, bd, N, 1.0, gd, N);
	}
}

void update_localX(const int N, const int *const restrict site_order, 
        const int nd, const int num_munu, const int l, const int L,
        const double dt, const double inv_dt_sq, uint64_t *const restrict rng,
		double *const restrict X,
		const int *const restrict map_i, const int *const restrict map_munu,
		const double *const restrict D, const int max_D_nums_nonzero,
		const int *const restrict D_nums_nonzero,
		const int *const restrict D_nonzero_inds,
		const double *const restrict local_box_widths,
		const int num_local_updates,
		const double *const restrict masses,
		struct meas_ph *const restrict m)
{
    double *dX = my_calloc(nd * sizeof(double));
	for (int ii = 0; ii < num_local_updates; ii++) {
		// printf("updates.c ln102, ii = %d\n", ii);
		const int i = site_order[ii];
		const int l_next = (l + 1) % L;
		const int l_prev = (l + L - 1) % L;
		double dEph = 0;
		const double pre = masses[map_i[i]] * inv_dt_sq;
		// printf("updates.c ln108, i = %d\n", i);
		for (int mu = 0; mu < nd; mu++) {
			const double dXmu = (rand_doub(rng) - 0.5) * local_box_widths[map_i[i]];
			dX[mu] = dXmu;
			dEph += dXmu * pre * (dXmu + 2*X[i + (mu + l*nd) * N]
			                             - X[i + (mu + l_next*nd) * N]
										 - X[i + (mu + l_prev*nd) * N]);
			// if (l == 0 && i ==1) {
			// 	printf("dXmu = %.12f\n", dXmu);
			// 	printf("pre = %.12f\n", pre);
			// 	printf("l_next = %d\n", l_next);
			// 	printf("l_prev = %d\n", l_prev);
			// 	printf("t-1 = %.12f\n", (dXmu + 2*X[i + (mu + l*nd) * N]
			//                              - X[i + (mu + l_next*nd) * N]
			// 							 - X[i + (mu + l_prev*nd) * N]));
			// 	printf("dEph = %.12f\n", dEph);
			// 	}
			for (int nu = 0; nu < nd; nu++) {
				const int munu = map_munu[nu + mu*nd];
				for (int jj = 0; jj < D_nums_nonzero[munu + i*num_munu]; jj++) {
					const int j = D_nonzero_inds[jj + (munu + i*num_munu) * max_D_nums_nonzero];
					dEph += dXmu * D[j + (i + munu*N) * N]
					             * (2 * X[j + (nu + l*nd) * N] + (mu == nu && i == j) * dXmu);
					// if (mu == nu && i == j) {
					// 	dEph += dXmu * D[j + (i + munu*N) * N] * (2 * X[j + (nu + l*nd) * N] + dXmu);
					// } else {
					// 	dEph += dXmu * D[j + (i + munu*N) * N] * X[j + (nu + l*nd) * N];
					// }
					// dEph += dXmu * D[j + (i + munu*N) * N] * X[j + (nu + l*nd) * N];
					// // if (l == 0 && i ==1) printf("dEph = %.12f\n", dEph);
					// if (mu == nu && i == j) {dEph += 0.5 * D[j + (i + munu*N) * N] * dXmu * dXmu;}
					// // if (l == 0 && i ==1) printf("dEph = %.12f\n", dEph);
				}
			}
		}
		m->n_local_total[map_i[i]]++;
		if (rand_doub(rng) < exp(-dt * dEph)) {
			m->n_local_accept[map_i[i]]++;
			for (int mu = 0; mu < nd; mu++) {X[i + (mu + l*nd) * N] += dX[mu];}
			// if (l == 0) {
				// printf("i = %d\n", i);
				// printf("dXmu = %.12f\n", dX[0]);
				// printf("dEph = %.12f\n", dEph);
			// }
		}
	}
	my_free(dX);
}

void update_blockX(const int N, const int *const restrict site_order,
        const int nd, const int num_munu, const int L, 
		const double dt, uint64_t *const restrict rng,
		double *const restrict X,
		const int *const restrict map_i, const int *const restrict map_munu,
		const double *const restrict D, const int max_D_nums_nonzero,
		const int *const restrict D_nums_nonzero,
		const int *const restrict D_nonzero_inds,
		const double *const restrict block_box_widths,
		const int num_block_updates,
		struct meas_ph *const restrict m)
{
	double *dX = my_calloc(nd * sizeof(double));
	for (int ii = 0; ii < num_block_updates; ii++) {
		const int i = site_order[ii];
		double dEph = 0;
		for (int mu = 0; mu < nd; mu++) {
			const double dXmu = (rand_doub(rng) - 0.5) * block_box_widths[map_i[i]];
			dX[mu] = dXmu;
			for (int nu = 0; nu < nd; nu++) {
				const int munu = map_munu[nu + mu*nd];
				for (int jj = 0; jj < D_nums_nonzero[munu + i*num_munu]; jj++) {
					const int j = D_nonzero_inds[jj + (munu + i*num_munu) * max_D_nums_nonzero];
					for (int l = 0; l < L; l++) {
						dEph += dXmu * D[j + (i + munu*N) * N]
						             * (2 * X[j + (nu + l*nd) * N] + (mu == nu && i == j) * dXmu);
						// if (mu == nu && i == j) {
						// 	dEph += dXmu * D[j + (i + munu*N) * N] * (2 * X[j + (nu + l*nd) * N] + dXmu);
						// } else {
						// 	dEph += dXmu * D[j + (i + munu*N) * N] * X[j + (nu + l*nd) * N];
						// }
						// dEph += dXmu * D[j + (i + munu*N) * N] * X[j + (nu + l*nd) * N];
						// if (mu == nu && i == j) {dEph += 0.5 * D[j + (i + munu*N) * N] * dXmu * dXmu;}
					}
				}
			}
		}
		m->n_block_total[map_i[i]]++;
		if (rand_doub(rng) < exp(-dt * dEph)) {
			m->n_block_accept[map_i[i]]++;
			for (int mu = 0; mu < nd; mu++) {
				for (int l = 0; l < L; l++) {
					X[i + (mu + l*nd) * N] += dX[mu];
				}
			}
			// printf("i = %d\n", i);
			// printf("dXmu = %.12f\n", dX[0]);
			// printf("dEph = %.12f\n", dEph);
		}
	}
	my_free(dX);
}

void update_flipX(const int N, const int *const restrict site_order,
        const int nd, const int num_munu, const int L,
		const double dt, uint64_t *const restrict rng,
		double *const restrict X,
		const int *const restrict map_i, const int *const restrict map_munu,
		const double *const restrict D, const int max_D_nums_nonzero,
		const int *const restrict D_nums_nonzero,
		const int *const restrict D_nonzero_inds,
		const int num_flip_updates,
		struct meas_ph *const restrict m)
{
	for (int ii = 0; ii < num_flip_updates; ii++) {
		const int i = site_order[ii];
		double dEph = 0;
		for (int l = 0; l < L; l++) {
			for (int mu = 0; mu < nd; mu++) {
				const double dXmul = -2 * X[i + (mu + l*nd) * N];
				for (int nu = 0; nu < nd; nu++) {
					const int munu = map_munu[nu + mu*nd];
					for (int jj = 0; jj < D_nums_nonzero[munu + i*num_munu]; jj++) {
						const int j = D_nonzero_inds[jj + (munu + i*num_munu) * max_D_nums_nonzero];
						dEph += dXmul * D[j + (i + munu*N) * N]
						              * (2 * X[j + (nu + l*nd) * N] + (mu == nu && i == j) * dXmul);
						// if (mu == nu && i == j) {
						// 	dEph += dXmul * D[j + (i + munu*N) * N] * (2 * X[j + (nu + l*nd) * N] + dXmul);
						// } else {
						// 	dEph += dXmul * D[j + (i + munu*N) * N] * X[j + (nu + l*nd) * N];
						// }
						// dEph += dXmul * D[j + (i + munu*N) * N] * X[j + (nu + l*nd) * N];
						// if (mu == nu && i == j) {dEph += 0.5 * D[j + (i + munu*N) * N] * dXmul * dXmul;}
					}
				}
			}
		}
		m->n_flip_total[map_i[i]]++;
		if (rand_doub(rng) < exp(-dt * dEph)) {
			m->n_flip_accept[map_i[i]]++;
			for (int l = 0; l < L; l++) {
				for (int mu = 0; mu < nd; mu++) {
					X[i + (mu + l*nd) * N] = -X[i + (mu + l*nd) * N];
				}
			}
			// printf("i = %d\n", i);
			// printf("dEph = %.12f\n", dEph);
		}
	}
}

// void update_localX(const int N, const int *const restrict site_order, const int l,
//         const double dt, const double inv_dt_sq, uint64_t *const restrict rng,
// 		double *const restrict X,
// 		double *const restrict exp_X, double *const restrict inv_exp_X,
// 		const double omega, const double omega_sq,
// 		const double local_box_width, const int num_local_updates,
// 		const double *const restrict gmat,
// 		const int max_num_coupledX,
// 		const int *const restrict num_coupledX,
// 		const int *const restrict coupledX_ind,
// 		num *const restrict gu, num *const restrict gd, num *const restrict phase,
// 		num *const restrict ggpu, num *const restrict cu, num *const restrict du,
// 		num *const restrict ggpd, num *const restrict cd, num *const restrict dd,
// 		int *const restrict pvtu, int *const restrict pvtd)
// {
// 	for (int ii = 0; ii < num_local_updates; ii++) {
// 		const int i = site_order[ii];
// 		const double dx = (rand_doub(rng) - 0.5) * local_box_width;
// 		const int n = num_coupledX[i];
// 		const int *const restrict X_inds = coupledX_ind + i * max_num_coupledX;
// 		const double *const restrict gm = gmat + i * N;
// 		for (int jj = 0; jj < n; jj++) {
// 			int j = X_inds[jj];
// 			for (int kk = 0; kk < n; kk++) {
// 				int k = X_inds[kk];
// 				int jk = j + k*N;
// 				if (j == k) {
// 					ggpu[jk] = 1 + (1 - gu[jk]) * expm1(-dt * gm[k] * dx);
// 					ggpd[jk] = 1 + (1 - gd[jk]) * expm1(-dt * gm[k] * dx);
// 				}
// 				else {
// 					ggpu[jk] = -gu[jk] * expm1(-dt * gm[k] * dx);
// 					ggpd[jk] = -gd[jk] * expm1(-dt * gm[k] * dx);
// 				}
// 			}
// 		}
// 		int infou, infod = 0;
// 		double ru, rd = 1;
// 		xgetrf(n, n, ggpu, N, pvtu, &infou);
// 		for (int j = 0; j < n; j++) {
// 			ru *= ggpu[j + j * n] * (pvtu[j] == (j + 1) ? 1 : -1);
// 		}
// 		xgetrf(n, n, ggpd, N, pvtd, &infod);
// 		for (int j = 0; j < n; j++) {
// 			rd *= ggpd[j + j * n] * (pvtd[j] == (j + 1) ? 1 : -1);
// 		}
// 	}
// }

/*
void update_shermor(const int N, const double *const restrict del,
		const int *const restrict site_order,
		uint64_t *const restrict rng, int *const restrict hs,
		double *const restrict gu, double *const restrict gd, int *const restrict sign,
		double *const restrict cu, double *const restrict du,
		double *const restrict cd, double *const restrict dd)
{
	for (int ii = 0; ii < N; ii++) {
		const int i = site_order[ii];
		const double delu = del[i + N*hs[i]];
		const double deld = del[i + N*!hs[i]];
		const double pu = (1 + (1 - gu[i + N*i])*delu);
		const double pd = (1 + (1 - gd[i + N*i])*deld);
		const double prob = pu*pd;
		if (rand_doub(rng) < fabs(prob)) {
			for (int j = 0; j < N; j++) cu[j] = gu[j + N*i];
			cu[i] -= 1.0;
			for (int j = 0; j < N; j++) du[j] = gu[i + N*j];
			const double au = delu/pu;
			dger(&N, &N, &au, cu, cint(1), du, cint(1), gu, &N);

			for (int j = 0; j < N; j++) cd[j] = gd[j + N*i];
			cd[i] -= 1.0;
			for (int j = 0; j < N; j++) dd[j] = gd[i + N*j];
			const double ad = deld/pd;
			dger(&N, &N, &ad, cd, cint(1), dd, cint(1), gd, &N);

			hs[i] = !hs[i];
			if (prob < 0) *sign *= -1;
		}
	}
}

void update_submat(const int N, const int q, const double *const restrict del,
		const int *const restrict site_order,
		uint64_t *const restrict rng, int *const restrict hs,
		double *const restrict gu, double *const restrict gd, int *const restrict sign,
		double *const restrict gr_u, double *const restrict g_ru,
		double *const restrict DDu, double *const restrict yu, double *const restrict xu,
		double *const restrict gr_d, double *const restrict g_rd,
		double *const restrict DDd, double *const restrict yd, double *const restrict xd)
{
	int *const restrict r = my_calloc(q * sizeof(int)); _aa(r);
	double *const restrict LUu = my_calloc(q*q * sizeof(double)); _aa(LUu);
	double *const restrict LUd = my_calloc(q*q * sizeof(double)); _aa(LUd);

	int k = 0;
	for (int ii = 0; ii < N; ii++) {
		const int i = site_order[ii];
		const double delu = del[i + N*hs[i]];
		const double deld = del[i + N*!hs[i]];
		double du = gu[i + N*i] - (1 + delu)/delu;
		double dd = gd[i + N*i] - (1 + deld)/deld;
		if (k > 0) {
			for (int j = 0; j < k; j++) yu[j] = gr_u[j + N*i];
			dtrtrs("L", "N", "U", &k, cint(1), LUu, &q, yu, &k, &(int){0});
			for (int j = 0; j < k; j++) xu[j] = g_ru[i + N*j];
			dtrtrs("U", "T", "N", &k, cint(1), LUu, &q, xu, &k, &(int){0});
			for (int j = 0; j < k; j++) du -= yu[j]*xu[j];

			for (int j = 0; j < k; j++) yd[j] = gr_d[j + N*i];
			dtrtrs("L", "N", "U", &k, cint(1), LUd, &q, yd, &k, &(int){0});
			for (int j = 0; j < k; j++) xd[j] = g_rd[i + N*j];
			dtrtrs("U", "T", "N", &k, cint(1), LUd, &q, xd, &k, &(int){0});
			for (int j = 0; j < k; j++) dd -= yd[j]*xd[j];
		}

		const double prob = du*delu * dd*deld;
		if (rand_doub(rng) < fabs(prob)) {
			r[k] = i;
			DDu[k] = 1.0 / (1.0 + delu);
			DDd[k] = 1.0 / (1.0 + deld);
			for (int j = 0; j < N; j++) gr_u[k + N*j] = gu[i + N*j];
			for (int j = 0; j < N; j++) g_ru[j + N*k] = gu[j + N*i];
			for (int j = 0; j < N; j++) gr_d[k + N*j] = gd[i + N*j];
			for (int j = 0; j < N; j++) g_rd[j + N*k] = gd[j + N*i];
			for (int j = 0; j < k; j++) LUu[j + q*k] = yu[j];
			for (int j = 0; j < k; j++) LUu[k + q*j] = xu[j];
			for (int j = 0; j < k; j++) LUd[j + q*k] = yd[j];
			for (int j = 0; j < k; j++) LUd[k + q*j] = xd[j];
			LUu[k + q*k] = du;
			LUd[k + q*k] = dd;
			k++;
			hs[i] = !hs[i];
			if (prob < 0) *sign *= -1;
		}

		if (k == q || (ii == N - 1 && k > 0)) {
			dtrtrs("L", "N", "U", &k, &N, LUu, &q, gr_u, &N, &(int){0});
			dtrtrs("U", "N", "N", &k, &N, LUu, &q, gr_u, &N, &(int){0});
			dgemm("N", "N", &N, &N, &k, cdbl(-1.0), g_ru, &N, gr_u, &N, cdbl(1.0), gu, &N);
			for (int j = 0; j < k; j++) {
				const int rj = r[j];
				const double DDk = DDu[j];
				for (int iii = 0; iii < N; iii++)
					gu[rj + iii*N] *= DDk;
			}
			dtrtrs("L", "N", "U", &k, &N, LUd, &q, gr_d, &N, &(int){0});
			dtrtrs("U", "N", "N", &k, &N, LUd, &q, gr_d, &N, &(int){0});
			dgemm("N", "N", &N, &N, &k, cdbl(-1.0), g_rd, &N, gr_d, &N, cdbl(1.0), gd, &N);
			for (int j = 0; j < k; j++) {
				const int rj = r[j];
				const double DDk = DDd[j];
				for (int iii = 0; iii < N; iii++)
					gd[rj + iii*N] *= DDk;
			}
			k = 0;
		}
	}
	my_free(LUd);
	my_free(LUu);
	my_free(r);
}
*/
