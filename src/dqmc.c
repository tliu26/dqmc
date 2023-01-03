#include "dqmc.h"
#include <tgmath.h>
#include <stdio.h>
#include "data.h"
#include "greens.h"
#include "linalg.h"
#include "meas.h"
#include "prof.h"
#include "rand.h"
#include "sig.h"
#include "time_.h"
#include "updates.h"
#include "util.h"
#include <stdarg.h>
#include <stdbool.h>

#define N_MUL 2 // input parameter to calc_eq_g() and calc_ue_g()

// uncomment to check recalculated g against wrapped g
// #define CHECK_G_WRP

// uncomment to check recalculated g against g from using QR for every multiply
// #define CHECK_G_ACC

// uncomment to check 0,0 block of unequal-time G against recalculated g
// #define CHECK_G_UE

// who needs function calls :D
#define matmul(C, A, B) do { \
	xgemm("N", "N", N, N, N, 1.0, (A), N, (B), N, 0.0, (C), N); \
} while (0);

#define calcBu(B, l) do { \
	for (int j = 0; j < N; j++) { \
		const double el = exp_lambda[j + N*hs[j + N*(l)]]; \
		for (int i = 0; i < N; i++) \
			(B)[i + N*j] = exp_Ku[i + N*j] * el; \
	} \
} while (0);

#define calcPhononBu(B, l) do { \
	for (int j = 0; j < N; j++) { \
		const double el = exp_lambda[j + N*hs[j + N*(l)]] * exp_X[j + N*(l)]; \
		for (int i = 0; i < N; i++) \
			(B)[i + N*j] = exp_Ku[i + N*j] * el; \
	} \
} while (0);

#define calcBd(B, l) do { \
	for (int j = 0; j < N; j++) { \
		const double el = exp_lambda[j + N*!hs[j + N*(l)]]; \
		for (int i = 0; i < N; i++) \
			(B)[i + N*j] = exp_Kd[i + N*j] * el; \
	} \
} while (0);

#define calcPhononBd(B, l) do { \
	for (int j = 0; j < N; j++) { \
		const double el = exp_lambda[j + N*!hs[j + N*(l)]] * exp_X[j + N*(l)]; \
		for (int i = 0; i < N; i++) \
			(B)[i + N*j] = exp_Kd[i + N*j] * el; \
	} \
} while (0);

#define calciBu(iB, l) do { \
	for (int i = 0; i < N; i++) { \
		const double el = exp_lambda[i + N*!hs[i + N*(l)]]; \
		for (int j = 0; j < N; j++) \
			(iB)[i + N*j] = el * inv_exp_Ku[i + N*j]; \
	} \
} while (0);

#define calcPhononiBu(iB, l) do { \
	for (int i = 0; i < N; i++) { \
		const double el = exp_lambda[i + N*!hs[i + N*(l)]] * inv_exp_X[i + N*(l)]; \
		for (int j = 0; j < N; j++) \
			(iB)[i + N*j] = el * inv_exp_Ku[i + N*j]; \
	} \
} while (0);

#define calciBd(iB, l) do { \
	for (int i = 0; i < N; i++) { \
		const double el = exp_lambda[i + N*hs[i + N*(l)]]; \
		for (int j = 0; j < N; j++) \
			(iB)[i + N*j] = el * inv_exp_Kd[i + N*j]; \
	} \
} while (0);

#define calcPhononiBd(iB, l) do { \
	for (int i = 0; i < N; i++) { \
		const double el = exp_lambda[i + N*hs[i + N*(l)]] * inv_exp_X[i + N*(l)]; \
		for (int j = 0; j < N; j++) \
			(iB)[i + N*j] = el * inv_exp_Kd[i + N*j]; \
	} \
} while (0);

#define matdiff(m, n, A, ldA, B, ldB) do { \
	double max = 0.0, avg = 0.0; \
	for (int j = 0; j < (n); j++) \
	for (int i = 0; i < (m); i++) { \
		const double diff = fabs((A)[i + (ldA)*j] - (B)[i + (ldB)*j]); \
		if (diff > max) max = diff; \
		avg += diff; \
	} \
	avg /= N*N; \
	printf(#A " - " #B ":\tmax %.3e\tavg %.3e\n", max, avg); \
} while (0);

void print_mat_f_dd(const double *const restrict mat, const char *mat_name, 
        int nd, const int *const ds, const int stride, 
		const int num_space)
{
	if (mat_name != NULL) {printf("%s = np.array(\n", mat_name);}
	printf("[");
	if (nd == 1) {
		int m = ds[0];
		for (int i = 0; i < m; i++) {
			printf("%.12f", mat[i*stride]);
			if (i < m-1) {printf(", ");}
		}
	} else {
		int nds[nd-1];
		for (int i = 0; i < nd-1; i++) {nds[i] = ds[i+1];}
		for (int i = 0; i < ds[0]; i++) {
			print_mat_f_dd(mat+i*stride, NULL, nd-1, nds, stride*ds[0],
			            num_space+1);
			if (i < ds[0]-1) {printf(",\n%*s", num_space, "");}
		}
	}
	printf("]");
	if (mat_name != NULL) {printf("\n)\n");}
}

void print_mat_f(const double *const restrict mat, const char *mat_name,
        int nd, ...)
{
	va_list valist;
	va_start(valist, nd);
	int ds[nd];
	for (int i = 0; i < nd; i++) {ds[i] = va_arg(valist, int);}
	print_mat_f_dd(mat, mat_name, nd, ds, 1, 1);
}

void print_mat_i_dd(const int *const restrict mat, const char *mat_name, 
        int nd, const int *const ds, const int stride, 
		const int num_space)
{
	if (mat_name != NULL) {printf("%s = np.array(\n", mat_name);}
	printf("[");
	if (nd == 1) {
		int m = ds[0];
		for (int i = 0; i < m; i++) {
			printf("%d", mat[i*stride]);
			if (i < m-1) {printf(", ");}
		}
	} else {
		int nds[nd-1];
		for (int i = 0; i < nd-1; i++) {nds[i] = ds[i+1];}
		for (int i = 0; i < ds[0]; i++) {
			print_mat_i_dd(mat+i*stride, NULL, nd-1, nds, stride*ds[0],
			            num_space+1);
			if (i < ds[0]-1) {printf(",\n%*s", num_space, "");}
		}
	}
	printf("]");
	if (mat_name != NULL) {printf("\n)\n");}
}

void print_mat_i(const int *const restrict mat, const char *mat_name,
        int nd, ...)
{
	va_list valist;
	va_start(valist, nd);
	int ds[nd];
	for (int i = 0; i < nd; i++) {ds[i] = va_arg(valist, int);}
	print_mat_i_dd(mat, mat_name, nd, ds, 1, 1);
}

int WriteData(const char *filename, const void *data, const size_t size,
              const size_t n, const bool append)
{
	const char *mode = append ? "ab" : "wb";
	FILE *fd = fopen(filename, mode);
	if (fd == NULL)
	{
		printf("'fopen()' failed during call of 'WriteData()'.\n");
		return -1;
	}
	// write data array to file
	if (fwrite(data, size, n, fd) != n)
	{
		printf("'fwrite()' failed during call of 'WriteData()'.\n");
		return -3;
	}
	fclose(fd);
	return 0;
}

static int dqmc(struct sim_data *sim)
{
	const int N = sim->p.N;
	const int L = sim->p.L;
	const int n_matmul = sim->p.n_matmul;
	const int n_delay = sim->p.n_delay;
	const int F = sim->p.F;
	const num *const restrict exp_Ku = sim->p.exp_Ku;
	const num *const restrict exp_Kd = sim->p.exp_Kd;
	const num *const restrict inv_exp_Ku = sim->p.inv_exp_Ku;
	const num *const restrict inv_exp_Kd = sim->p.inv_exp_Kd;
	const num *const restrict exp_halfKu = sim->p.exp_halfKu;
	const num *const restrict exp_halfKd = sim->p.exp_halfKd;
	const num *const restrict inv_exp_halfKu = sim->p.inv_exp_halfKu;
	const num *const restrict inv_exp_halfKd = sim->p.inv_exp_halfKd;
	const double *const restrict exp_lambda = sim->p.exp_lambda;
	const double *const restrict del = sim->p.del;
	uint64_t *const restrict rng = sim->s.rng;
	int *const restrict hs = sim->s.hs;
	const double dt = sim->p.dt;
	const double inv_dt_sq = sim->p.inv_dt_sq;
	const int *const restrict map_munu = sim->php.map_munu;
	const int max_D_nums_nonzero = sim->php.max_D_nums_nonzero;
	const int nd = sim->php.nd;
	const int num_munu = sim->php.num_munu;
	const double *const restrict D = sim->php.D;
	// const double *const restrict gmat = sim->php.gmat;
	// const int *const restrict num_coupledX = sim->php.num_coupledX;
	// const int *const restrict coupledX_ind = sim->php.coupledX_ind;
	const int *const restrict D_nums_nonzero = sim->php.D_nums_nonzero;
	const int *const restrict D_nonzero_inds = sim->php.D_nonzero_inds;
	double *const restrict X = sim->s.X;
	// double *const restrict exp_X = sim->s.exp_X;
	// double *const restrict inv_exp_X = sim->s.inv_exp_X;

	num *const Bu = my_calloc(N*N*L * sizeof(num));
	num *const Bd = my_calloc(N*N*L * sizeof(num));
	num *const iBu = my_calloc(N*N*L * sizeof(num));
	num *const iBd = my_calloc(N*N*L * sizeof(num));
	num *const Cu = my_calloc(N*N*F * sizeof(num));
	num *const Cd = my_calloc(N*N*F * sizeof(num));
	num *const restrict gu = my_calloc(N*N * sizeof(num));
	num *const restrict gd = my_calloc(N*N * sizeof(num));
	#ifdef CHECK_G_WRP
	num *const restrict guwrp = my_calloc(N*N * sizeof(num));
	num *const restrict gdwrp = my_calloc(N*N * sizeof(num));
	#endif
	#ifdef CHECK_G_ACC
	num *const restrict guacc = my_calloc(N*N * sizeof(num));
	num *const restrict gdacc = my_calloc(N*N * sizeof(num));
	#endif
	num phase;
	int *const site_order = my_calloc(N * sizeof(double));

	// work arrays for calc_eq_g and stuff. two sets for easy 2x parallelization
	num *const restrict tmpNN1u = my_calloc(N*N * sizeof(num));
	num *const restrict tmpNN2u = my_calloc(N*N * sizeof(num));
	num *const restrict tmpN1u = my_calloc(N * sizeof(num));
	num *const restrict tmpN2u = my_calloc(N * sizeof(num));
	num *const restrict tmpN3u = my_calloc(N * sizeof(num));
	int *const restrict pvtu = my_calloc(N * sizeof(int));

	num *const restrict tmpNN1d = my_calloc(N*N * sizeof(num));
	num *const restrict tmpNN2d = my_calloc(N*N * sizeof(num));
	num *const restrict tmpN1d = my_calloc(N * sizeof(num));
	num *const restrict tmpN2d = my_calloc(N * sizeof(num));
	num *const restrict tmpN3d = my_calloc(N * sizeof(num));
	int *const restrict pvtd = my_calloc(N * sizeof(int));

	// arrays for calc_ue_g
	num *restrict Gu0t = NULL;
	num *restrict Gutt = NULL;
	num *restrict Gut0 = NULL;
	// num *restrict ueGu = NULL;
	num *restrict Gredu = NULL;
	num *restrict tauu = NULL;
	num *restrict Qu = NULL;

	num *restrict Gd0t = NULL;
	num *restrict Gdtt = NULL;
	num *restrict Gdt0 = NULL;
	// num *restrict ueGd = NULL;
	num *restrict Gredd = NULL;
	num *restrict taud = NULL;
	num *restrict Qd = NULL;

	if (sim->p.period_uneqlt > 0) {
		const int E = 1 + (F - 1) / N_MUL;

		Gredu = my_calloc(N*E*N*E * sizeof(num));
		tauu = my_calloc(N*E * sizeof(num));
		Qu = my_calloc(4*N*N * sizeof(num));

		Gredd = my_calloc(N*E*N*E * sizeof(num));
		taud = my_calloc(N*E * sizeof(num));
		Qd = my_calloc(4*N*N * sizeof(num));

		Gu0t = my_calloc(N*N*L * sizeof(num));
		Gutt = my_calloc(N*N*L * sizeof(num));
		Gut0 = my_calloc(N*N*L * sizeof(num));
		Gd0t = my_calloc(N*N*L * sizeof(num));
		Gdtt = my_calloc(N*N*L * sizeof(num));
		Gdt0 = my_calloc(N*N*L * sizeof(num));
		// ueGu = my_calloc(N*N*L*L * sizeof(num));
		// ueGd = my_calloc(N*N*L*L * sizeof(num));
		// if (ueGu == NULL || ueGd == NULL) return -1;
	}

	// lapack work arrays
	int lwork = get_lwork_eq_g(N);
	if (sim->p.period_uneqlt > 0) {
		const int E = 1 + (F - 1) / N_MUL;
		const int lwork_ue = get_lwork_ue_g(N, E);
		if (lwork_ue > lwork) lwork = lwork_ue;
	}
	num *const restrict worku = my_calloc(lwork * sizeof(num));
	num *const restrict workd = my_calloc(lwork * sizeof(num));

	{
	num phaseu, phased;
	#pragma omp parallel sections
	{
	#pragma omp section
	{
	if (sim->p.period_uneqlt > 0)
		for (int l = 0; l < L; l++)
			calciBu(iBu + N*N*l, l);
	for (int l = 0; l < L; l++)
		calcBu(Bu + N*N*l, l);
	for (int f = 0; f < F; f++)
		mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L, 1.0,
		        Bu, Cu + N*N*f, N, tmpNN1u);
	phaseu = calc_eq_g(0, N, F, N_MUL, Cu, gu, tmpNN1u, tmpNN2u,
	                  tmpN1u, tmpN2u, tmpN3u, pvtu, worku, lwork);
	}
	#pragma omp section
	{
	if (sim->p.period_uneqlt > 0)
		for (int l = 0; l < L; l++)
			calciBd(iBd + N*N*l, l);
	for (int l = 0; l < L; l++)
		calcBd(Bd + N*N*l, l);
	for (int f = 0; f < F; f++)
		mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L, 1.0,
		        Bd, Cd + N*N*f, N, tmpNN1d);
	phased = calc_eq_g(0, N, F, N_MUL, Cd, gd, tmpNN1d, tmpNN2d,
	                  tmpN1d, tmpN2d, tmpN3d, pvtd, workd, lwork);
	}
	}
	phase = phaseu*phased;
	}

	// print_mat_f(exp_Ku, "exp_Ku", 2, N, N);
	// print_mat_f(exp_lambda, "exp_lambda(s=0)", 2, 1, N);
	// print_mat_i(hs, "hs[l=0]", 2, 1, N);
	// print_mat_f(Bu, "Bu[l=0]", 2, N, N);
	// print_mat_f(gmat, "gmat", 2, N, N);
	// print_mat_f(D, "D", 3, N, N, 6);
	// print_mat_f(X, "X", 3, N, 3, L);
	// print_mat_f(exp_X, "exp_X", 2, N, L);
	// print_mat_i(num_coupledX, "num_coupledX", 2, 1, N);
	// print_mat_i(coupledX_ind, "coupledX_ind", 2, sim->php.max_num_coupledX, N);
	// print_mat_i(sim->php.D_nums_nonzero, "D_nums_nonzero", 2, 6, N);
	// print_mat_i(sim->php.D_nonzero_inds, "D_nonzero_inds", 3, sim->php.max_D_nums_nonzero, 6, N);
	// printf("dqmc.c ln363\n");
	for (; sim->s.sweep < sim->p.n_sweep; sim->s.sweep++) {
		const int sig = sig_check_state(sim->s.sweep, sim->p.n_sweep_warm, sim->p.n_sweep);
		if (sig == 1) // stop flag
			break;
		else if (sig == 2) { // progress flag
			const int status = sim_data_save(sim);
			if (status < 0)
				fprintf(stderr, "save_file() failed: %d\n", status);
		}

		for (int l = 0; l < L; l++) {
			profile_begin(updates);
			shuffle(rng, N, site_order);
			update_delayed(N, n_delay, del, site_order,
			               rng, hs + N*l, gu, gd, &phase,
			               tmpNN1u, tmpNN2u, tmpN1u,
			               tmpNN1d, tmpNN2d, tmpN1d);
			profile_end(updates);

			profile_begin(localX_update);
			shuffle(rng, N, site_order);
			// if (l == 0) print_mat_f(X, "X0", 2, N, L);
			// printf("l = %d\n", l);
			// print_mat_f(X, "X0", 2, N, L);
			update_localX(N, site_order, nd, num_munu, l, L, dt, inv_dt_sq, rng,
			              X, sim->p.map_i, map_munu,
						  D, max_D_nums_nonzero, D_nums_nonzero, D_nonzero_inds,
						  sim->php.local_box_widths, sim->php.num_local_updates,
						  sim->php.masses,
						  &sim->m_ph);
			// if (l == 0) print_mat_f(X, "X1", 2, N, L);
			// print_mat_f(X, "X1", 2, N, L);
			profile_end(localX_update);

			const int f = l / n_matmul;
			const int recalc = ((l + 1) % n_matmul == 0);
			num phaseu, phased;
			#pragma omp parallel sections
			{
			#pragma omp section
			{
			num *const restrict Bul = Bu + N*N*l;
			num *const restrict iBul = iBu + N*N*l;
			num *const restrict Cuf = Cu + N*N*f;
			profile_begin(calcb);
			calcBu(Bul, l);
			if (!recalc || sim->p.period_uneqlt > 0)
				calciBu(iBul, l);
			profile_end(calcb);
			if (recalc) {
				profile_begin(multb);
				mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L,
				        1.0, Bu, Cuf, N, tmpNN1u);
				profile_end(multb);
				profile_begin(recalc);
				#ifdef CHECK_G_WRP
				if (sim->p.period_uneqlt == 0)
					calciBu(iBu + N*N*l, l);
				matmul(tmpNN1u, gu, iBu + N*N*l);
				matmul(guwrp, Bu + N*N*l, tmpNN1u);
				#endif
				#ifdef CHECK_G_ACC
				calc_eq_g((l + 1) % L, N, L, 1, Bu, guacc,
				          tmpNN1u, tmpNN2u, tmpN1u, tmpN2u,
				          tmpN3u, pvtu, worku, lwork);
				#endif
				phaseu = calc_eq_g((f + 1) % F, N, F, N_MUL, Cu, gu,
				                  tmpNN1u, tmpNN2u, tmpN1u, tmpN2u,
				                  tmpN3u, pvtu, worku, lwork);
				profile_end(recalc);
			} else {
				profile_begin(wrap);
				matmul(tmpNN1u, gu, iBul);
				matmul(gu, Bul, tmpNN1u);
				profile_end(wrap);
			}
			}
			#pragma omp section
			{
			num *const restrict Bdl = Bd + N*N*l;
			num *const restrict iBdl = iBd + N*N*l;
			num *const restrict Cdf = Cd + N*N*f;
			profile_begin(calcb);
			calcBd(Bdl, l);
			if (!recalc || sim->p.period_uneqlt > 0)
				calciBd(iBdl, l);
			profile_end(calcb);
			if (recalc) {
				profile_begin(multb);
				mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L,
				        1.0, Bd, Cdf, N, tmpNN1d);
				profile_end(multb);
				profile_begin(recalc);
				#ifdef CHECK_G_WRP
				if (sim->p.period_uneqlt == 0)
					calciBd(iBd + N*N*l, l);
				matmul(tmpNN1d, gd, iBd + N*N*l);
				matmul(gdwrp, Bd + N*N*l, tmpNN1d);
				#endif
				#ifdef CHECK_G_ACC
				calc_eq_g((l + 1) % L, N, L, 1, Bd, gdacc,
				          tmpNN1d, tmpNN2d, tmpN1d, tmpN2d,
				          tmpN3d, pvtd, workd, lwork);
				#endif
				phased = calc_eq_g((f + 1) % F, N, F, N_MUL, Cd, gd,
				                  tmpNN1d, tmpNN2d, tmpN1d, tmpN2d,
				                  tmpN3d, pvtd, workd, lwork);
				profile_end(recalc);
			} else {
				profile_begin(wrap);
				matmul(tmpNN1d, gd, iBdl);
				matmul(gd, Bdl, tmpNN1d);
				profile_end(wrap);
			}
			}
			}

			#ifdef CHECK_G_WRP
			if (recalc) {
				matdiff(N, N, gu, N, guwrp, N);
				matdiff(N, N, gd, N, gdwrp, N);
			}
			#endif
			#ifdef CHECK_G_ACC
			if (recalc) {
				matdiff(N, N, gu, N, guacc, N);
				matdiff(N, N, gd, N, gdacc, N);
			}
			#endif
			#if defined(CHECK_G_WRP) && defined(CHECK_G_ACC)
			if (recalc) {
				matdiff(N, N, guwrp, N, guacc, N);
				matdiff(N, N, gdwrp, N, gdacc, N);
			}
			#endif

			if (recalc) phase = phaseu*phased;

            // printf("sim->s.sweep = %d, l = %d\n", sim->s.sweep, l);
			if ((sim->s.sweep >= sim->p.n_sweep_warm) &&
					(sim->p.period_eqlt > 0) &&
					(l + 1) % sim->p.period_eqlt == 0) {
				#pragma omp parallel sections
				{
				#pragma omp section
				{
				profile_begin(half_wrap);
				matmul(tmpNN1u, gu, exp_halfKu);
				matmul(tmpNN2u, inv_exp_halfKu, tmpNN1u);
				profile_end(half_wrap);
				}
				#pragma omp section
				{
				profile_begin(half_wrap);
				matmul(tmpNN1d, gd, exp_halfKd);
				matmul(tmpNN2d, inv_exp_halfKd, tmpNN1d);
				profile_end(half_wrap);
				}
				}

				profile_begin(meas_eq);
				measure_eqlt(&sim->p, phase, tmpNN2u, tmpNN2d, &sim->m_eq);
				profile_end(meas_eq);
			}
		}

        // print_mat_f(X, "X0", 2, N, L);
		profile_begin(blockX_update);
		shuffle(rng, N, site_order);
		update_blockX(N, site_order, nd, num_munu, L, dt, rng,
		              X, sim->p.map_i, map_munu, D,
		              max_D_nums_nonzero, D_nums_nonzero, D_nonzero_inds,
					  sim->php.block_box_widths, sim->php.num_block_updates,
					  &sim->m_ph);
		profile_end(blockX_update);
		// print_mat_f(X, "X1", 2, N, L);

        // print_mat_f(X, "X0", 2, N, L);
		profile_begin(flipX_update);
		shuffle(rng, N, site_order);
		update_flipX(N, site_order, nd, num_munu, L, dt, rng,
		              X, sim->p.map_i, map_munu, D,
		              max_D_nums_nonzero, D_nums_nonzero, D_nonzero_inds,
					  sim->php.num_flip_updates, &sim->m_ph);
		profile_end(flipX_update);
		// print_mat_f(X, "X1", 2, N, L);

		if (sim->php.track_phonon_ite > 0) {
			char path[1024];
			for (int mu = 0; mu < nd; mu++) {
				for (int i = 0; i < N; i++) {
					sprintf(path, "phonon_ite_Xl0i%d.dat", i);
					WriteData(path, X + i + mu*N, sizeof(double), 1, true);
				}
			}
		}

		if ((sim->s.sweep >= sim->p.n_sweep_warm) && (sim->p.period_uneqlt > 0) &&
				sim->s.sweep % sim->p.period_uneqlt == 0) {
			#pragma omp parallel sections
			{
			#pragma omp section
			calc_ue_g(N, L, F, N_MUL, Bu, iBu, Cu, Gu0t, Gutt, Gut0,
			          Gredu, tauu, Qu, worku, lwork);
			#pragma omp section
			calc_ue_g(N, L, F, N_MUL, Bd, iBd, Cd, Gd0t, Gdtt, Gdt0,
			          Gredd, taud, Qd, workd, lwork);
			}

			#ifdef CHECK_G_UE
			matdiff(N, N, gu, N, Gutt, N);
			matdiff(N, N, gd, N, Gdtt, N);
			#endif
			#if defined(CHECK_G_UE) && defined(CHECK_G_ACC)
			matdiff(N, N, Gutt, N, guacc, N);
			matdiff(N, N, Gdtt, N, gdacc, N);
			#endif

			#pragma omp parallel sections
			{
			#pragma omp section
			{
			profile_begin(half_wrap);
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1u, Gu0t + N*N*l, exp_halfKu);
				matmul(Gu0t + N*N*l, inv_exp_halfKu, tmpNN1u);
			}
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1u, Gutt + N*N*l, exp_halfKu);
				matmul(Gutt + N*N*l, inv_exp_halfKu, tmpNN1u);
			}
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1u, Gut0 + N*N*l, exp_halfKu);
				matmul(Gut0 + N*N*l, inv_exp_halfKu, tmpNN1u);
			}
			profile_end(half_wrap);
			}
			#pragma omp section
			{
			profile_begin(half_wrap);
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1d, Gd0t + N*N*l, exp_halfKd);
				matmul(Gd0t + N*N*l, inv_exp_halfKd, tmpNN1d);
			}
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1d, Gdtt + N*N*l, exp_halfKd);
				matmul(Gdtt + N*N*l, inv_exp_halfKd, tmpNN1d);
			}
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1d, Gdt0 + N*N*l, exp_halfKd);
				matmul(Gdt0 + N*N*l, inv_exp_halfKd, tmpNN1d);
			}
			profile_end(half_wrap);
			}
			}

			profile_begin(meas_uneq);
			measure_uneqlt(&sim->p, phase,
			               Gu0t, Gutt, Gut0, Gd0t, Gdtt, Gdt0,
			               &sim->m_ue);
			profile_end(meas_uneq);
			// #pragma omp parallel sections
			// {
			// #pragma omp section
			// calc_ue_g(N, L, F, N_MUL, Bu, iBu, Cu,
			          // ueGu, Gredu, tauu, Qu, worku, lwork);
			// #pragma omp section
			// calc_ue_g(N, L, F, N_MUL, Bd, iBd, Cd,
			          // ueGd, Gredd, taud, Qd, workd, lwork);
			// }

			// #ifdef CHECK_G_UE
			// matdiff(N, N, gu, N, ueGu, N);
			// matdiff(N, N, gd, N, ueGd, N);
			// #endif
			// #if defined(CHECK_G_UE) && defined(CHECK_G_ACC)
			// matdiff(N, N, ueGu, N, guacc, N);
			// matdiff(N, N, ueGd, N, gdacc, N);
			// #endif

			// profile_begin(meas_uneq);
			// measure_uneqlt(&sim->p, sign, ueGu, ueGd, &sim->m_ue);
			// profile_end(meas_uneq);
		}
		if (sim->s.sweep >= sim->p.n_sweep_warm) {
			profile_begin(meas_phonon);
			measure_ph(&sim->p, phase, nd, X, &sim->m_ph);
			profile_end(meas_phonon);
		}
	}


	my_free(workd);
	my_free(worku);
	if (sim->p.period_uneqlt > 0) {
		my_free(Qd);
		my_free(taud);
		my_free(Gredd);
		// my_free(ueGd);
		my_free(Gdt0);
		my_free(Gdtt);
		my_free(Gd0t);
		my_free(Qu);
		my_free(tauu);
		my_free(Gredu);
		// my_free(ueGu);
		my_free(Gut0);
		my_free(Gutt);
		my_free(Gu0t);
	}
	my_free(pvtd);
	my_free(tmpN3d);
	my_free(tmpN2d);
	my_free(tmpN1d);
	my_free(tmpNN2d);
	my_free(tmpNN1d);
	my_free(pvtu);
	my_free(tmpN3u);
	my_free(tmpN2u);
	my_free(tmpN1u);
	my_free(tmpNN2u);
	my_free(tmpNN1u);
	my_free(site_order);
	#ifdef CHECK_G_ACC
	my_free(gdacc);
	my_free(guacc);
	#endif
	#ifdef CHECK_G_WRP
	my_free(gdwrp);
	my_free(guwrp);
	#endif
	my_free(gd);
	my_free(gu);
	my_free(Cd);
	my_free(Cu);
	my_free(iBd);
	my_free(iBu);
	my_free(Bd);
	my_free(Bu);

	return 0;
}

int dqmc_wrapper(const char *sim_file, const char *log_file,
		const tick_t max_time, const int bench)
{
	const tick_t wall_start = time_wall();
	profile_clear();

	int status = 0;

	// open log file
	FILE *log = (log_file != NULL) ? fopen(log_file, "a") : stdout;
	if (log == NULL) {
		fprintf(stderr, "fopen() failed to open: %s\n", log_file);
		return -1;
	}

	fprintf(log, "commit id %s\n", GIT_ID);
	fprintf(log, "compiled on %s %s\n", __DATE__, __TIME__);

	// initialize signal handling
	sig_init(log, wall_start, max_time);

	// open and read simulation file
	struct sim_data *sim = my_calloc(sizeof(struct sim_data));
	fprintf(log, "opening %s\n", sim_file);
	status = sim_data_read_alloc(sim, sim_file);
	if (status < 0) {
		fprintf(stderr, "read_file() failed: %d\n", status);
		status = -1;
		goto cleanup;
	}

	// check existing progress
	fprintf(log, "%d/%d sweeps completed\n", sim->s.sweep, sim->p.n_sweep);
	if (sim->s.sweep >= sim->p.n_sweep) {
		fprintf(log, "already finished\n");
		goto cleanup;
	}

	// run dqmc
	fprintf(log, "starting dqmc\n");
	status = dqmc(sim);
	if (status < 0) {
		fprintf(stderr, "dqmc() failed to allocate memory\n");
		status = -1;
		goto cleanup;
	}
	fprintf(log, "%d/%d sweeps completed\n", sim->s.sweep, sim->p.n_sweep);

	// save to simulation file (if not in benchmarking mode)
	if (!bench) {
		fprintf(log, "saving data\n");
		status = sim_data_save(sim);
		if (status < 0) {
			fprintf(stderr, "save_file() failed: %d\n", status);
			status = -1;
			goto cleanup;
		}
	} else {
		fprintf(log, "benchmark mode enabled; not saving data\n");
	}

	status = (sim->s.sweep == sim->p.n_sweep) ? 0 : 1;

cleanup:
	sim_data_free(sim);
	my_free(sim);

	const tick_t wall_time = time_wall() - wall_start;
	fprintf(log, "wall time: %.3f\n", wall_time * SEC_PER_TICK);
	profile_print(log, wall_time);

	if (log != stdout)
		fclose(log);
	else
		fflush(log);

	return status;
}
