#pragma once

#include <stdint.h>
#include "util.h"

struct params {
	int N, L;
	int *map_i, *map_ij;
	int *bonds, *map_bs, *map_bb;
	num *peierlsu, *peierlsd;
//	double *K, *U;
	double dt, inv_dt_sq;
	double chem_pot;

	int n_matmul, n_delay;
	int n_sweep_warm, n_sweep_meas;
	int period_eqlt, period_uneqlt;
	int meas_bond_corr, meas_energy_corr, meas_nematic_corr;

	int num_i, num_ij;
	int num_b, num_bs, num_bb;
	int *degen_i, *degen_ij, *degen_bs, *degen_bb;
	num *exp_Ku, *exp_Kd, *inv_exp_Ku, *inv_exp_Kd;
	num *exp_halfKu, *exp_halfKd, *inv_exp_halfKu, *inv_exp_halfKd;
	double *exp_lambda, *del;
	int F, n_sweep;
};

struct phonon_params {
	int nd, num_munu;
	double *D;
	double *ks;
	int max_D_nums_nonzero;
	int *D_nums_nonzero;
	int *D_nonzero_inds;
	int *map_munu;
	double *local_box_widths, *block_box_widths;
	int num_local_updates, num_block_updates, num_flip_updates;
	double *masses;
	double *gmat;
	int max_num_coupled_to_X;
	int *num_coupled_to_X;
	int *ind_coupled_to_X;
	int track_phonon_ite;
};

struct state {
	uint64_t rng[17];
	int sweep;
	int *hs;
	double *X;
	double *exp_X;
	double *inv_exp_X;
};

struct meas_eqlt {
	int n_sample;
	num sign;

	num *density;
	num *double_occ;

	num *g00;
	num *nn;
	num *xx;
	num *zz;
	num *pair_sw;
	num *kk, *kv, *kn, *vv, *vn;
};

struct meas_uneqlt {
	int n_sample;
	num sign;

	num *gt0;
	num *nn;
	num *xx;
	num *zz;
	num *pair_sw;
	num *pair_bb;
	num *jj, *jsjs;
	num *kk, *ksks;
	num *kv, *kn, *vv, *vn;
	num *nem_nnnn, *nem_ssss;
};

struct meas_ph {
	int *n_local_accept;
	int *n_local_total;
	int *n_block_accept;
	int *n_block_total;
	int *n_flip_accept;
	int *n_flip_total;
	int n_sample;
	num sign;

	num *X_avg;
	num *X_avg_sq;
	num *X_sq_avg;
	num *X_cubed_avg;
	num *V_avg;
	num *V_sq_avg;
	num *PE;
	num *KE;
	num *nX;

	num *XX;
};

struct sim_data {
	const char *file;
	struct params p;
	struct phonon_params php;
	struct state s;
	struct meas_eqlt m_eq;
	struct meas_uneqlt m_ue;
	struct meas_ph m_ph;
};

int sim_data_read_alloc(struct sim_data *sim, const char *file);

int sim_data_save(const struct sim_data *sim);

void sim_data_free(const struct sim_data *sim);
