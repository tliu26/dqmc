#include "updates.h"
#include <tgmath.h>
#include "linalg.h"
#include "rand.h"
#include "util.h"
#include "greens.h"
#include <stdio.h>
#include <stdarg.h>

#define N_MUL 2

static inline double qed(const double x)
{
	return square(x) * square(x);
}

void calcPBu(num *const Bu, const int l, const int N,
             const double *const restrict exp_lambda, const int *const restrict hs,
			 const double *const restrict exp_X, const num *const restrict exp_Ku)
{
	for (int j = 0; j < N; j++) {
		const double el = exp_lambda[j + N*hs[j + N*l]] * exp_X[j + N*l];
		for (int i = 0; i < N; i++)
			Bu[i + N*j] = exp_Ku[i + N*j] * el;
	}
}

void calcPBd(num *const Bd, const int l, const int N,
             const double *const restrict exp_lambda, const int *const restrict hs,
			 const double *const restrict exp_X, const num *const restrict exp_Kd)
{
	for (int j = 0; j < N; j++) {
		const double el = exp_lambda[j + N*!hs[j + N*l]] * exp_X[j + N*l];
		for (int i = 0; i < N; i++)
			Bd[i + N*j] = exp_Kd[i + N*j] * el;
	}
}

void calcPiBu(num *const iBu, const int l, const int N,
              const double *const restrict exp_lambda, const int *const restrict hs,
			  const double *const restrict inv_exp_X, const num *const restrict inv_exp_Ku)
{
	for (int i = 0; i < N; i++) {
		const double el = exp_lambda[i + N*!hs[i + N*l]] * inv_exp_X[i + N*l];
		for (int j = 0; j < N; j++)
			iBu[i + N*j] = el * inv_exp_Ku[i + N*j];
	}
}

void calcPiBd(num *const iBd, const int l, const int N,
              const double *const restrict exp_lambda, const int *const restrict hs,
			  const double *const restrict inv_exp_X, const num *const restrict inv_exp_Kd)
{
	for (int i = 0; i < N; i++) {
		const double el = exp_lambda[i + N*hs[i + N*l]] * inv_exp_X[i + N*l];
		for (int j = 0; j < N; j++)
			iBd[i + N*j] = el * inv_exp_Kd[i + N*j];
	}
}

void print_mat_f_dd_rowmaj_updates(const double *const mat, const char *mat_name,
        int nd, const int *const ds, const int num_space)
{
	if (mat_name != NULL) {printf("%s = np.array(\n", mat_name);}
	printf("[");
	if (nd == 1) {
		int m = ds[0];
		for (int i = 0; i < m; i++) {
			printf("%.18G", mat[i]);
			if (i < m-1) {printf(", ");}
		}
	} else {
		int nds[nd-1];
		int stride = 1;
		for (int i = 0; i < nd-1; i++) {nds[i] = ds[i+1]; stride *= ds[i+1];}
		for (int i = 0; i < ds[0]; i++) {
			print_mat_f_dd_rowmaj_updates(mat+i*stride, NULL, nd-1, nds, num_space+1);
			if (i < ds[0]-1) {printf(",\n%*s", num_space, "");}
		}
	}
	printf("]");
	if (mat_name != NULL) {printf("\n)\n");}
}

void print_mat_f_rowmaj_updates(const double *const mat, const char *mat_name,
        int nd, ...)
{
	va_list valist;
	va_start(valist, nd);
	int ds[nd];
	for (int i = 0; i < nd; i++) {ds[i] = va_arg(valist, int);}
	print_mat_f_dd_rowmaj_updates(mat, mat_name, nd, ds, 1);
}

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

void update_sherman_morrison(const int N, const int i, const double del,
        num *const restrict g, num *const restrict c, num *const restrict d)
{
	for (int j = 0; j < N; j++) c[j] = g[j + N*i];
	c[i] -= 1.0;
	for (int j = 0; j < N; j++) d[j] = g[i + N*j];
	const num a = del/(1 - del*c[i]);
	xgeru(N, N, a, c, 1, d, 1, g, N);
}

void update_localX(const int N, const int *const restrict site_order,
        const int nd, const int num_munu, const int num_i,
		const int l, const int L,
        const double dt, const double inv_dt_sq, uint64_t *const restrict rng,
		double *const restrict X,
		double *const restrict exp_X, double *const restrict inv_exp_X,
		num *const restrict gu, num *const restrict gd, num *const restrict phase,
		num *const restrict ggpu, num *const restrict cu, num *const restrict du,
		int *const restrict pvtu,
		num *const restrict ggpd, num *const restrict cd, num *const restrict dd,
		int *const restrict pvtd,
		const int *const restrict map_i, const int *const restrict map_munu,
		const double *const restrict D, const int max_D_nums_nonzero,
		const int *const restrict D_nums_nonzero,
		const int *const restrict D_nonzero_inds,
		const double *const restrict ks,
		const double *const gmat, const int n_max,
		const int *const restrict num_coupled_to_X,
		const int *const ind_coupled_to_X,
		const double *const restrict local_box_widths,
		const int num_local_updates,
		const double *const restrict masses,
		struct meas_ph *const restrict m)
{
	// printf("l = %d\n", l);
    double *dX = my_calloc(nd * sizeof(double));
	for (int ii = 0; ii < num_local_updates; ii++) {
		const int i = site_order[ii];
		const int l_next = (l + 1) % L;
		const int l_prev = (l + L - 1) % L;
		double dEph = 0;
		const double pre = masses[map_i[i]] * inv_dt_sq;
		const int n = num_coupled_to_X[i];
		const int *const js = ind_coupled_to_X + i * n_max;
		double *delta_hep = my_calloc(n * sizeof(double));
		double *exp_delta_hep = my_calloc(n * sizeof(double));
		for (int mu = 0; mu < nd; mu++) {
			const double dXmu = (rand_doub(rng) - 0.5) * local_box_widths[map_i[i]];
			dX[mu] = dXmu;
			// phonon action
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
				}
			}
            dEph += ks[map_i[i] + mu*num_i] *
			        (qed(X[i + (mu + l*nd) * N] + dXmu) - qed(X[i + (mu + l*nd) * N]));
			// el-ph interaction
			const double *const gm = gmat + (i + mu * N) * N;
			for (int jj = 0; jj < n; jj++) {
				int j = js[jj];
				delta_hep[jj] += gm[j] * dXmu;
			}
		}
		// determinant due to change in el-ph interaction
		for (int jj = 0; jj < n; jj++) {
			int j = js[jj];
			exp_delta_hep[jj] = exp(-dt * delta_hep[jj]);
			for (int kk = 0; kk < n; kk++) {
				int k = js[kk];
				int kkjj = kk + jj*n;
				int kj = k + j*N;
				if (j == k) {
					ggpu[kkjj] = 1 + (1 - gu[kj]) * (exp_delta_hep[jj]-1);
					ggpd[kkjj] = 1 + (1 - gd[kj]) * (exp_delta_hep[jj]-1);
				} else {
					ggpu[kkjj] = -gu[kj] * (exp_delta_hep[jj]-1);
					ggpd[kkjj] = -gd[kj] * (exp_delta_hep[jj]-1);
				}
			}
		}
		// printf("before update:\n");
		// print_mat_f_rowmaj_updates(gu, "gu", 2, N, N);
		// print_mat_f_rowmaj_updates(gd, "gd", 2, N, N);
		int infou, infod;
		num ru = 1;
		num rd = 1;
		xgetrf(n, n, ggpu, n, pvtu, &infou);
		xgetrf(n, n, ggpd, n, pvtd, &infod);
		for (int j = 0; j < n; j++) {
			ru *= ggpu[j + j * n] * (pvtu[j] == (j + 1) ? 1 : -1);
			rd *= ggpd[j + j * n] * (pvtd[j] == (j + 1) ? 1 : -1);
		}
		// if (l == 10) {
		// 	printf("l = %d\n", l);
		// 	printf("i = %d\n", i);
		// 	print_mat_f_rowmaj_updates(dX, "dX", 1, nd);
		// 	printf("ru = %.12f\n", ru);
		// 	printf("rd = %.12f\n", rd);
		// }

		m->n_local_total[map_i[i]]++;
		const num prob = ru * rd * exp(-dt * dEph);
		const double absprob = fabs(prob);
		if (rand_doub(rng) < absprob) {
			// printf("l = %d, accepted\n", l);
			m->n_local_accept[map_i[i]]++;
			// update phonon field
			for (int mu = 0; mu < nd; mu++) {X[i + (mu + l*nd) * N] += dX[mu];}
			for (int jj = 0; jj < n; jj++) {
				int j = js[jj];
				exp_X[j + l*N] *= exp_delta_hep[jj];
				inv_exp_X[j + l*N] /= exp_delta_hep[jj];
			}
			// if (l == 0) {
				// printf("i = %d\n", i);
				// printf("dXmu = %.12f\n", dX[0]);
				// printf("dEph = %.12f\n", dEph);
			// }
			// update electron Green's function with Sherman-Morrison
			for (int jj = 0; jj < n; jj++) {
			    update_sherman_morrison(N, js[jj], exp_delta_hep[jj]-1, gu, cu, du);
				update_sherman_morrison(N, js[jj], exp_delta_hep[jj]-1, gd, cd, dd);
			}
			*phase *= prob/absprob;
			// printf("accepted\n");
			// printf("ru = %.18G\n", ru);
			// printf("rd = %.18G\n", rd);
			// print_mat_f_rowmaj_updates(gu, "gu_p", 2, N, N);
			// print_mat_f_rowmaj_updates(gd, "gd_p", 2, N, N);
			// printf("\n");
		}
		my_free(delta_hep);
		my_free(exp_delta_hep);
	}
	my_free(dX);
	// printf("\n");
}

void update_blockX(const int N, const int *const restrict site_order,
        const int nd, const int num_munu, const int num_i,
		const int L, const int F, const int n_matmul,
		const double dt, uint64_t *const restrict rng,
		double *const restrict X, double *const restrict exp_X, double *const restrict inv_exp_X,
		const double *const restrict exp_lambda, const int *const restrict hs,
		const num *const restrict exp_Ku, const num *const restrict exp_Kd,
		double *const restrict logdetu, double *const restrict logdetd,
		num *const restrict tmpNN1u, num *const restrict tmpNN2u,
		num *const restrict tmpN1u, num *const restrict tmpN2u, num *const restrict tmpN3u,
		int *const restrict pvtu, num *const restrict worku,
		num *const restrict tmpNN1d, num *const restrict tmpNN2d,
		num *const restrict tmpN1d, num *const restrict tmpN2d, num *const restrict tmpN3d,
		int *const restrict pvtd, num *const restrict workd, const int lwork,
		const int *const restrict map_i, const int *const restrict map_munu,
		const double *const restrict D, const int max_D_nums_nonzero,
		const int *const restrict D_nums_nonzero, const int *const restrict D_nonzero_inds,
		const double *const restrict ks,
		const double *const gmat, const int n_max,
		const int *const restrict num_coupled_to_X,
		const int *const ind_coupled_to_X,
		const double *const restrict block_box_widths,
		const int num_block_updates,
		struct meas_ph *const restrict m)
{
	double *dX = my_calloc(nd * sizeof(double));
	double *const restrict tmp_exp_X = my_calloc(L*N*sizeof(double));
	num *const tmp_Bu = my_calloc(N*N*L*sizeof(num));
	num *const tmp_Bd = my_calloc(N*N*L*sizeof(num));
	num *const tmp_Cu = my_calloc(N*N*F*sizeof(num));
	num *const tmp_Cd = my_calloc(N*N*F*sizeof(num));
	num *const restrict tmp_gu = my_calloc(N*N*L*sizeof(num));
	num *const restrict tmp_gd = my_calloc(N*N*L*sizeof(num));
	double logdetu_p, logdetd_p;
	num phaseu_p, phased_p;
	int infou, infod;

	for (int ii = 0; ii < num_block_updates; ii++) {
		const int i = site_order[ii];
		double dEph = 0;
		const int n = num_coupled_to_X[i];
		const int *const js = ind_coupled_to_X + i * n_max;
		memcpy(tmp_exp_X, exp_X, L*N*sizeof(double));
		for (int mu = 0; mu < nd; mu++) {
			// phonon action
			const double dXmu = (rand_doub(rng) - 0.5) * block_box_widths[map_i[i]];
			dX[mu] = dXmu;
			for (int nu = 0; nu < nd; nu++) {
				const int munu = map_munu[nu + mu*nd];
				for (int jj = 0; jj < D_nums_nonzero[munu + i*num_munu]; jj++) {
					const int j = D_nonzero_inds[jj + (munu + i*num_munu) * max_D_nums_nonzero];
					for (int l = 0; l < L; l++)
						dEph += dXmu * D[j + (i + munu*N) * N]
						             * (2 * X[j + (nu + l*nd) * N] + (mu == nu && i == j) * dXmu);
				}
			}
			for (int l = 0; l < L; l++)
			    dEph += ks[map_i[i] + mu*num_i] *
				        (qed(X[i + (mu + l*nd) * N] + dXmu) - qed(X[i + (mu + l*nd) * N]));
			// el-ph coupling
			const double *const gm = gmat + (i + mu * N) * N;
			for (int jj = 0; jj < n; jj++) {
				int j = js[jj];
				for (int l = 0; l < L; l++)
					tmp_exp_X[j + l*N] *= exp(-dt * gm[j] * dXmu);
			}
		}
		logdetu_p = 0;
		logdetd_p = 0;
		#pragma omp parallel sections
		{
		#pragma omp section
		{
		for (int l = 0; l < L; l++)
			calcPBu(tmp_Bu + N*N*l, l, N, exp_lambda, hs, tmp_exp_X, exp_Ku);
		for (int f = 0; f < F; f++)
			mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L, 1.0,
					tmp_Bu, tmp_Cu + N*N*f, N, tmpNN1u);
		phaseu_p = calc_eq_g(0, N, F, N_MUL, tmp_Cu, tmp_gu,
		                     tmpNN1u, tmpNN2u, tmpN1u, tmpN2u,
							 tmpN3u, pvtu, worku, lwork);
		for (int i = 0; i < N; i++)
			logdetu_p += log(fabs(tmpN3u[i])) - log(fabs(tmpNN2u[i + i*N]));
		}
		#pragma omp section
		{
		for (int l = 0; l < L; l++)
			calcPBd(tmp_Bd + N*N*l, l, N, exp_lambda, hs, tmp_exp_X, exp_Kd);
		for (int f = 0; f < F; f++)
			mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L, 1.0,
					tmp_Bd, tmp_Cd + N*N*f, N, tmpNN1d);
		phased_p = calc_eq_g(0, N, F, N_MUL, tmp_Cd, tmp_gd,
		                     tmpNN1d, tmpNN2d, tmpN1d, tmpN2d,
							 tmpN3d, pvtd, workd, lwork);
		for (int i = 0; i < N; i++)
			logdetd_p += log(fabs(tmpN3d[i])) - log(fabs(tmpNN2d[i + i*N]));
		}
		}

		// print_mat_f_rowmaj_updates(tmp_gu, "tmp_gu", 2, N, N);
		// print_mat_f_rowmaj_updates(tmp_gd, "tmp_gd", 2, N, N);
		// printf("logdetu_p = %.18G\n", logdetu_p);
		// printf("phaseu_p = %.18G\n", phaseu_p);
		// printf("logdetd_p = %.18G\n", logdetd_p);
		// printf("phased_p = %.18G\n", phased_p);

        // double logdetd_plu = 0;
		// xgetrf(N, N, tmp_gd, N, pvtd, &infod);
		// for (int i = 0; i < N; i++)
		// 	logdetd_plu += log(fabs(tmp_gd[i + i * N]));
		// printf("logdetd_plu = %.18G\n", logdetd_plu);

		m->n_block_total[map_i[i]]++;
		if (rand_doub(rng) < exp((*logdetu + *logdetd) - (logdetu_p + logdetd_p) - dt * dEph)) {
			m->n_block_accept[map_i[i]]++;
			// for (int mu = 0; mu < nd; mu++) {
			// 	for (int l = 0; l < L; l++) {
			// 		X[i + (mu + l*nd) * N] += dX[mu];
			// 	}
			// }
			for (int l = 0; l < L; l++) {
				for (int mu = 0; mu < nd; mu++)
					X[i + (mu + l*nd) * N] += dX[mu];
				for (int jj = 0; jj < n; jj++) {
					int j = js[jj];
					exp_X[j + l*N] = tmp_exp_X[j + l*N];
					inv_exp_X[j + l*N] = 1 / tmp_exp_X[j + l*N];
				}
			}
			*logdetu = logdetu_p;
			*logdetd = logdetd_p;
			// printf("i = %d\n", i);
			// printf("dXmu = %.12f\n", dX[0]);
			// printf("dEph = %.12f\n", dEph);
		}
	}
	my_free(dX);
	my_free(tmp_exp_X);
	my_free(tmp_Bu);
	my_free(tmp_Bd);
	my_free(tmp_Cu);
	my_free(tmp_Cd);
	my_free(tmp_gu);
	my_free(tmp_gd);
}

void update_flipX(const int N, const int *const restrict site_order,
        const int nd, const int num_munu, const int num_i,
		const int L, const int F, const int n_matmul,
		const double dt, const double chem_pot, uint64_t *const restrict rng,
		double *const restrict X, double *const restrict exp_X, double *const restrict inv_exp_X,
		const double *const restrict exp_lambda, const int *const restrict hs,
		const num *const restrict exp_Ku, const num *const restrict exp_Kd,
		double *const restrict logdetu, double *const restrict logdetd,
		num *const restrict tmpNN1u, num *const restrict tmpNN2u,
		num *const restrict tmpN1u, num *const restrict tmpN2u, num *const restrict tmpN3u,
		int *const restrict pvtu, num *const restrict worku,
		num *const restrict tmpNN1d, num *const restrict tmpNN2d,
		num *const restrict tmpN1d, num *const restrict tmpN2d, num *const restrict tmpN3d,
		int *const restrict pvtd, num *const restrict workd, const int lwork,
		const int *const restrict map_i, const int *const restrict map_munu,
		const double *const restrict D, const int max_D_nums_nonzero,
		const int *const restrict D_nums_nonzero,
		const int *const restrict D_nonzero_inds,
		const double *const restrict ks,
		const double *const gmat, const int n_max,
		const int *const restrict num_coupled_to_X,
		const int *const ind_coupled_to_X,
		const int num_flip_updates,
		struct meas_ph *const restrict m)
{
	double *const restrict tmp_exp_X = my_calloc(L*N*sizeof(double));
	num *const tmp_Bu = my_calloc(N*N*L*sizeof(num));
	num *const tmp_Bd = my_calloc(N*N*L*sizeof(num));
	num *const tmp_Cu = my_calloc(N*N*F*sizeof(num));
	num *const tmp_Cd = my_calloc(N*N*F*sizeof(num));
	num *const restrict tmp_gu = my_calloc(N*N*L*sizeof(num));
	num *const restrict tmp_gd = my_calloc(N*N*L*sizeof(num));
	double *const restrict dX = my_calloc(L*nd*sizeof(double));
	double logdetu_p, logdetd_p;
	num phaseu_p, phased_p;
	int infou, infod;

	for (int ii = 0; ii < num_flip_updates; ii++) {
		const int i = site_order[ii];
		double dEph = 0;
		const int n = num_coupled_to_X[i];
		const int *const js = ind_coupled_to_X + i * n_max;
		memcpy(tmp_exp_X, exp_X, L*N*sizeof(double));
		for (int l = 0; l < L; l++) {
			for (int mu = 0; mu < nd; mu++) {
				const double *const gm = gmat + (i + mu * N) * N;
				double flip = 0;
				for (int jj = 0; jj < n; jj++) flip += gm[js[jj]];
				if (flip != 0) flip = 2 * chem_pot / flip;
				// printf("flip = %.18G\n", flip);
				const double dXmul = flip - 2 * X[i + (mu + l*nd) * N];
				dX[mu + l*nd] = dXmul;
				// phonon action
				for (int nu = 0; nu < nd; nu++) {
					const int munu = map_munu[nu + mu*nd];
					for (int jj = 0; jj < D_nums_nonzero[munu + i*num_munu]; jj++) {
						const int j = D_nonzero_inds[jj + (munu + i*num_munu) * max_D_nums_nonzero];
						dEph += dXmul * D[j + (i + munu*N) * N]
						              * (2 * X[j + (nu + l*nd) * N] + (mu == nu && i == j) * dXmul);
					}
				}
			    dEph += ks[map_i[i] + mu*num_i] *
				        (qed(X[i + (mu + l*nd) * N] + dXmul) - qed(X[i + (mu + l*nd) * N]));
				// el-ph coupling
				for (int jj = 0; jj < n; jj++) {
					int j = js[jj];
					tmp_exp_X[j + l*N] *= exp(-dt * gm[j] * dXmul);
				}
			}
		}
		logdetu_p = 0;
		logdetd_p = 0;
		#pragma omp parallel sections
		{
		#pragma omp section
		{
		for (int l = 0; l < L; l++)
			calcPBu(tmp_Bu + N*N*l, l, N, exp_lambda, hs, tmp_exp_X, exp_Ku);
		for (int f = 0; f < F; f++)
			mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L, 1.0,
					tmp_Bu, tmp_Cu + N*N*f, N, tmpNN1u);
		phaseu_p = calc_eq_g(0, N, F, N_MUL, tmp_Cu, tmp_gu,
		                     tmpNN1u, tmpNN2u, tmpN1u, tmpN2u,
							 tmpN3u, pvtu, worku, lwork);
		for (int i = 0; i < N; i++)
			logdetu_p += log(fabs(tmpN3u[i])) - log(fabs(tmpNN2u[i + i*N]));
		}
		#pragma omp section
		{
		for (int l = 0; l < L; l++)
			calcPBd(tmp_Bd + N*N*l, l, N, exp_lambda, hs, tmp_exp_X, exp_Kd);
		for (int f = 0; f < F; f++)
			mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L, 1.0,
					tmp_Bd, tmp_Cd + N*N*f, N, tmpNN1d);
		phased_p = calc_eq_g(0, N, F, N_MUL, tmp_Cd, tmp_gd,
		                     tmpNN1d, tmpNN2d, tmpN1d, tmpN2d,
							 tmpN3d, pvtd, workd, lwork);
		for (int i = 0; i < N; i++)
			logdetd_p += log(fabs(tmpN3d[i])) - log(fabs(tmpNN2d[i + i*N]));
		}
		}
		m->n_flip_total[map_i[i]]++;
		if (rand_doub(rng) < exp((*logdetu + *logdetd) - (logdetu_p + logdetd_p) - dt * dEph)) {
			m->n_flip_accept[map_i[i]]++;
			for (int l = 0; l < L; l++) {
				for (int mu = 0; mu < nd; mu++)
					X[i + (mu + l*nd) * N] += dX[mu + l*nd];
				for (int jj = 0; jj < n; jj++) {
					int j = js[jj];
					exp_X[j + l*N] = tmp_exp_X[j + l*N];
					inv_exp_X[j + l*N] = 1 / tmp_exp_X[j + l*N];
				}
			}
			*logdetu = logdetu_p;
			*logdetd = logdetd_p;
			// printf("i = %d\n", i);
			// printf("dEph = %.12f\n", dEph);
		}
	}
	my_free(tmp_exp_X);
	my_free(tmp_Bu);
	my_free(tmp_Bd);
	my_free(tmp_Cu);
	my_free(tmp_Cd);
	my_free(tmp_gu);
	my_free(tmp_gd);
	my_free(dX);
}

void update_time_step_mats(const int N, const int L, const int F, const int n_matmul,
        const double *const restrict exp_X, const double *const restrict inv_exp_X,
        const double *const restrict exp_lambda, const int *const restrict hs,
		const num *const restrict exp_Ku, const num *const restrict exp_Kd,
		const num *const restrict inv_exp_Ku, const num *const restrict inv_exp_Kd,
		num *const Bu, num *const iBu, num *const restrict gu,
		num *const Bd, num *const iBd, num *const restrict gd,
		num *const Cu, num *const Cd, num *const restrict phase,
		num *const restrict tmpNN1u, num *const restrict tmpNN2u,
		num *const restrict tmpN1u, num *const restrict tmpN2u, num *const restrict tmpN3u,
		int *const restrict pvtu, num *const restrict worku,
		num *const restrict tmpNN1d, num *const restrict tmpNN2d,
		num *const restrict tmpN1d, num *const restrict tmpN2d, num *const restrict tmpN3d,
		int *const restrict pvtd, num *const restrict workd, const int lwork)
{
	num phaseu, phased;
	#pragma omp parallel sections
	{
	#pragma omp section
	{
	for (int l = 0; l < L; l++) {
		calcPBu(Bu + N*N*l, l, N, exp_lambda, hs, exp_X, exp_Ku);
		calcPiBu(iBu + N*N*l, l, N, exp_lambda, hs, inv_exp_X, inv_exp_Ku);
	}
	for (int f = 0; f < F; f++)
		mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L, 1.0,
		        Bu, Cu + N*N*f, N, tmpNN1u);
	phaseu = calc_eq_g(0, N, F, N_MUL, Cu, gu,
	                   tmpNN1u, tmpNN2u, tmpN1u, tmpN2u,
					   tmpN3u, pvtu, worku, lwork);
	}
	#pragma omp section
	{
	for (int l = 0; l < L; l++) {
		calcPBd(Bd + N*N*l, l, N, exp_lambda, hs, exp_X, exp_Kd);
		calcPiBd(iBd + N*N*l, l, N, exp_lambda, hs, inv_exp_X, inv_exp_Kd);
	}
	for (int f = 0; f < F; f++)
		mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L, 1.0,
		        Bd, Cd + N*N*f, N, tmpNN1d);
	phased = calc_eq_g(0, N, F, N_MUL, Cd, gd,
	                   tmpNN1d, tmpNN2d, tmpN1d, tmpN2d,
					   tmpN3d, pvtd, workd, lwork);
	}
	}
	*phase = phaseu*phased;
}

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
