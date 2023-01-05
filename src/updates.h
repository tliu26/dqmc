#pragma once

#include <stdint.h>
#include "util.h"
#include "data.h"

void update_delayed(const int N, const int n_delay, const double *const restrict del,
		const int *const restrict site_order,
		uint64_t *const restrict rng, int *const restrict hs,
		num *const restrict Gu, num *const restrict Gd, num *const restrict phase,
		// work arrays (sizes: N*N, N*N, N)
		num *const restrict au, num *const restrict bu, num *const restrict du,
		num *const restrict ad, num *const restrict bd, num *const restrict dd);

void update_sherman_morrison(const int N, const int i, const double del,
        num *const restrict g, num *const restrict c, num *const restrict d);

void update_localX(const int N, const int *const restrict site_order,
        const int nd, const int num_munu, const int l, const int L,
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
		const double *const gmat, const int n_max,
		const int *const restrict num_coupled_to_X,
		const int *const ind_coupled_to_X,
		const double *const restrict local_box_widths,
		const int num_local_updates,
		const double *const restrict masses,
		struct meas_ph *const restrict m);

void update_blockX(const int N, const int *const restrict site_order,
        const int nd, const int num_munu, const int L, const int F, const int n_matmul,
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
		const double *const gmat, const int n_max,
		const int *const restrict num_coupled_to_X,
		const int *const ind_coupled_to_X,
		const double *const restrict block_box_widths,
		const int num_block_updates,
		struct meas_ph *const restrict m);

void update_flipX(const int N, const int *const restrict site_order,
        const int nd, const int num_munu, const int L, const int F, const int n_matmul,
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
		const double *const gmat, const int n_max,
		const int *const restrict num_coupled_to_X,
		const int *const ind_coupled_to_X,
		const int num_flip_updates,
		struct meas_ph *const restrict m);

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
		int *const restrict pvtd, num *const restrict workd, const int lwork);

/*
// regular sherman morrison
void update_shermor(const int N, const double *const restrict del,
		const int *const restrict site_order,
		uint64_t *const restrict rng, int *const restrict hs,
		double *const restrict Gu, double *const restrict Gd, int *const restrict sign,
		// work arrays (sizes: N, N)
		double *const restrict cu, double *const restrict du,
		double *const restrict cd, double *const restrict dd);

// submatrix updates. generally worse performance except for very large N (>1000)
void update_submat(const int N, const int q, const double *const restrict del,
		const int *const restrict site_order,
		uint64_t *const restrict rng, int *const restrict hs,
		double *const restrict Gu, double *const restrict Gd, int *const restrict sign,
		// work arrays (sizes: N*N, N*N, N, N, N)
		double *const restrict Gr_u, double *const restrict G_ru,
		double *const restrict DDu, double *const restrict yu, double *const restrict xu,
		double *const restrict Gr_d, double *const restrict G_rd,
		double *const restrict DDd, double *const restrict yd, double *const restrict xd);
*/
