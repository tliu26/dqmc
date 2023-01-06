#include "data.h"
#include <stdio.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include "util.h"

#define return_if(cond, val, ...) \
	do {if (cond) {fprintf(stderr, __VA_ARGS__); return (val);}} while (0)

static hid_t num_h5t;

int sim_data_read_alloc(struct sim_data *sim, const char *file)
{
	const hid_t file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
	return_if(file_id < 0, -1, "H5Fopen() failed: %ld\n", file_id);

	sim->file = file;

	herr_t status;

#ifdef USE_CPLX
	num_h5t = H5Tcreate(H5T_COMPOUND, sizeof(num));
	status = H5Tinsert(num_h5t, "r", 0, H5T_NATIVE_DOUBLE);
	return_if(status < 0, -1, "H5Tinsert() failed: %d\n", status);
	status = H5Tinsert(num_h5t, "i", 8, H5T_NATIVE_DOUBLE);
	return_if(status < 0, -1, "H5Tinsert() failed: %d\n", status);
#else
	num_h5t = H5T_NATIVE_DOUBLE;
#endif


#define my_read(_type, name, ...) do { \
	status = H5LTread_dataset##_type(file_id, (name), __VA_ARGS__); \
	return_if(status < 0, -1, "H5LTread_dataset() failed for %s: %d\n", (name), status); \
} while (0);

	my_read(_int, "/params/N",      &sim->p.N);
	my_read(_int, "/params/L",      &sim->p.L);
	my_read(_double, "/params/dt", &sim->p.dt);
	my_read(_double, "/params/inv_dt_sq", &sim->p.inv_dt_sq);
	my_read(_double, "/params/chem_pot", &sim->p.chem_pot);
	my_read(_int, "/params/num_i",  &sim->p.num_i);
	my_read(_int, "/params/num_ij", &sim->p.num_ij);
	my_read(_int, "/params/num_b", &sim->p.num_b);
	my_read(_int, "/params/num_bs", &sim->p.num_bs);
	my_read(_int, "/params/num_bb", &sim->p.num_bb);
	my_read(_int, "/params/period_uneqlt", &sim->p.period_uneqlt);
	my_read(_int, "/params/meas_bond_corr", &sim->p.meas_bond_corr);
	my_read(_int, "/params/meas_energy_corr", &sim->p.meas_energy_corr);
	my_read(_int, "/params/meas_nematic_corr", &sim->p.meas_nematic_corr);
	my_read(_double, "/params/phonon_k", &sim->php.phonon_k);
	my_read(_int, "/params/nd", &sim->php.nd);
	my_read(_int, "/params/num_munu", &sim->php.num_munu);
	my_read(_int, "/params/max_D_nums_nonzero", &sim->php.max_D_nums_nonzero);
	my_read(_int, "/params/max_num_coupled_to_X",   &sim->php.max_num_coupled_to_X);
	my_read(_int, "/params/num_local_updates",  &sim->php.num_local_updates);
	my_read(_int, "/params/num_block_updates",  &sim->php.num_block_updates);
	my_read(_int, "/params/num_flip_updates",  &sim->php.num_flip_updates);
	my_read(_int, "/params/track_phonon_ite", &sim->php.track_phonon_ite);

	const int N = sim->p.N, L = sim->p.L;
	const int num_i = sim->p.num_i, num_ij = sim->p.num_ij;
	const int num_b = sim->p.num_b, num_bs = sim->p.num_bs, num_bb = sim->p.num_bb;
	const int nd = sim->php.nd;
	const int num_munu = sim->php.num_munu;
	const int max_D_nums_nonzero = sim->php.max_D_nums_nonzero;
	const int max_num_coupled_to_X = sim->php.max_num_coupled_to_X;

	sim->p.map_i         = my_calloc(N        * sizeof(int));
	sim->p.map_ij        = my_calloc(N*N      * sizeof(int));
	sim->p.bonds         = my_calloc(num_b*2  * sizeof(int));
	sim->p.map_bs        = my_calloc(num_b*N  * sizeof(int));
	sim->p.map_bb        = my_calloc(num_b*num_b * sizeof(int));
	sim->p.peierlsu      = my_calloc(N*N      * sizeof(num));
	sim->p.peierlsd      = my_calloc(N*N      * sizeof(num));
//	sim->p.K             = my_calloc(N*N      * sizeof(double));
//	sim->p.U             = my_calloc(num_i    * sizeof(double));
	sim->p.degen_i       = my_calloc(num_i    * sizeof(int));
	sim->p.degen_ij      = my_calloc(num_ij   * sizeof(int));
	sim->p.degen_bs      = my_calloc(num_bs   * sizeof(int));
	sim->p.degen_bb      = my_calloc(num_bb   * sizeof(int));
	sim->p.exp_Ku        = my_calloc(N*N      * sizeof(num));
	sim->p.exp_Kd        = my_calloc(N*N      * sizeof(num));
	sim->p.inv_exp_Ku    = my_calloc(N*N      * sizeof(num));
	sim->p.inv_exp_Kd    = my_calloc(N*N      * sizeof(num));
	sim->p.exp_halfKu    = my_calloc(N*N      * sizeof(num));
	sim->p.exp_halfKd    = my_calloc(N*N      * sizeof(num));
	sim->p.inv_exp_halfKu= my_calloc(N*N      * sizeof(num));
	sim->p.inv_exp_halfKd= my_calloc(N*N      * sizeof(num));
	sim->p.exp_lambda    = my_calloc(N*2      * sizeof(double));
	sim->p.del           = my_calloc(N*2      * sizeof(double));
	sim->php.map_munu      = my_calloc(nd * nd * sizeof(int));
	sim->php.D             = my_calloc(num_munu * N * N * sizeof(double));
	sim->php.D_nums_nonzero = my_calloc(num_munu * N * sizeof(int));
	sim->php.D_nonzero_inds = my_calloc(num_munu * N * max_D_nums_nonzero * sizeof(int));
	sim->php.local_box_widths = my_calloc(num_i * sizeof(double));
	sim->php.block_box_widths = my_calloc(num_i * sizeof(double));
	sim->php.masses = my_calloc(num_i * sizeof(double));
	sim->php.gmat          = my_calloc(nd*N*N      * sizeof(double));
	sim->php.num_coupled_to_X = my_calloc(N        * sizeof(int));
	sim->php.ind_coupled_to_X = my_calloc(N*max_num_coupled_to_X * sizeof(int));
	sim->s.hs            = my_calloc(N*L      * sizeof(int));
	sim->s.X             = my_calloc(N*nd*L    * sizeof(double));
	sim->s.exp_X         = my_calloc(N*L      * sizeof(double));
	sim->s.inv_exp_X     = my_calloc(N*L      * sizeof(double));
	sim->m_eq.density    = my_calloc(num_i    * sizeof(num));
	sim->m_eq.double_occ = my_calloc(num_i    * sizeof(num));
	sim->m_eq.g00        = my_calloc(num_ij   * sizeof(num));
	sim->m_eq.nn         = my_calloc(num_ij   * sizeof(num));
	sim->m_eq.xx         = my_calloc(num_ij   * sizeof(num));
	sim->m_eq.zz         = my_calloc(num_ij   * sizeof(num));
	sim->m_eq.pair_sw    = my_calloc(num_ij   * sizeof(num));
	if (sim->p.meas_energy_corr) {
		sim->m_eq.kk = my_calloc(num_bb * sizeof(num));
		sim->m_eq.kv = my_calloc(num_bs * sizeof(num));
		sim->m_eq.kn = my_calloc(num_bs * sizeof(num));
		sim->m_eq.vv = my_calloc(num_ij * sizeof(num));
		sim->m_eq.vn = my_calloc(num_ij * sizeof(num));
	}
	if (sim->p.period_uneqlt > 0) {
		sim->m_ue.gt0     = my_calloc(num_ij*L * sizeof(num));
		sim->m_ue.nn      = my_calloc(num_ij*L * sizeof(num));
		sim->m_ue.xx      = my_calloc(num_ij*L * sizeof(num));
		sim->m_ue.zz      = my_calloc(num_ij*L * sizeof(num));
		sim->m_ue.pair_sw = my_calloc(num_ij*L * sizeof(num));
		if (sim->p.meas_bond_corr) {
			sim->m_ue.pair_bb = my_calloc(num_bb*L * sizeof(num));
			sim->m_ue.jj      = my_calloc(num_bb*L * sizeof(num));
			sim->m_ue.jsjs    = my_calloc(num_bb*L * sizeof(num));
			sim->m_ue.kk      = my_calloc(num_bb*L * sizeof(num));
			sim->m_ue.ksks    = my_calloc(num_bb*L * sizeof(num));
		}
		if (sim->p.meas_energy_corr) {
			sim->m_ue.kv      = my_calloc(num_bs*L * sizeof(num));
			sim->m_ue.kn      = my_calloc(num_bs*L * sizeof(num));
			sim->m_ue.vv      = my_calloc(num_ij*L * sizeof(num));
			sim->m_ue.vn      = my_calloc(num_ij*L * sizeof(num));
		}
		if (sim->p.meas_nematic_corr) {
			sim->m_ue.nem_nnnn = my_calloc(num_bb*L * sizeof(num));
			sim->m_ue.nem_ssss = my_calloc(num_bb*L * sizeof(num));
		}
	}
	sim->m_ph.X_avg = my_calloc(nd*num_i * sizeof(num));
	sim->m_ph.X_avg_sq = my_calloc(nd*num_i * sizeof(num));
	sim->m_ph.X_sq_avg = my_calloc(nd*num_i * sizeof(num));
	sim->m_ph.X_cubed_avg = my_calloc(nd*num_i * sizeof(num));
	sim->m_ph.V_avg = my_calloc(nd*num_i * sizeof(num));
	sim->m_ph.V_sq_avg = my_calloc(nd*num_i * sizeof(num));
	sim->m_ph.PE = my_calloc(nd*num_i * sizeof(num));
	sim->m_ph.KE = my_calloc(nd*num_i * sizeof(num));
	sim->m_ph.nX = my_calloc(nd*num_ij*L * sizeof(num));
	sim->m_ph.XX = my_calloc(nd*nd*num_ij*L * sizeof(num));
	sim->m_ph.n_local_accept = my_calloc(num_i * sizeof(int));
	sim->m_ph.n_local_total = my_calloc(num_i * sizeof(int));
	sim->m_ph.n_block_accept = my_calloc(num_i * sizeof(int));
	sim->m_ph.n_block_total = my_calloc(num_i * sizeof(int));
	sim->m_ph.n_flip_accept = my_calloc(num_i * sizeof(int));
	sim->m_ph.n_flip_total = my_calloc(num_i * sizeof(int));
	// make sure anything appended here is free'd in sim_data_free()

	my_read(_int,    "/params/map_i",          sim->p.map_i);
	my_read(_int,    "/params/map_ij",         sim->p.map_ij);
	my_read(_int,    "/params/bonds",          sim->p.bonds);
	my_read(_int,    "/params/map_bs",         sim->p.map_bs);
	my_read(_int,    "/params/map_bb",         sim->p.map_bb);
	my_read( , "/params/peierlsu", num_h5t,    sim->p.peierlsu);
	my_read( , "/params/peierlsd", num_h5t,    sim->p.peierlsd);
//	my_read(_double, "/params/K",              sim->p.K);
//	my_read(_double, "/params/U",              sim->p.U);
//	my_read(_double, "/params/dt",            &sim->p.dt);
	my_read(_int,    "/params/n_matmul",      &sim->p.n_matmul);
	my_read(_int,    "/params/n_delay",       &sim->p.n_delay);
	my_read(_int,    "/params/n_sweep_warm",  &sim->p.n_sweep_warm);
	my_read(_int,    "/params/n_sweep_meas",  &sim->p.n_sweep_meas);
	my_read(_int,    "/params/period_eqlt",   &sim->p.period_eqlt);
	my_read(_int,    "/params/degen_i",        sim->p.degen_i);
	my_read(_int,    "/params/degen_ij",       sim->p.degen_ij);
	my_read(_int,    "/params/degen_bs",       sim->p.degen_bs);
	my_read(_int,    "/params/degen_bb",       sim->p.degen_bb);
	my_read( , "/params/exp_Ku",     num_h5t,   sim->p.exp_Ku);
	my_read( , "/params/exp_Kd",     num_h5t,   sim->p.exp_Kd);
	my_read( , "/params/inv_exp_Ku", num_h5t,   sim->p.inv_exp_Ku);
	my_read( , "/params/inv_exp_Kd", num_h5t,   sim->p.inv_exp_Kd);
	my_read( , "/params/exp_halfKu",     num_h5t,   sim->p.exp_halfKu);
	my_read( , "/params/exp_halfKd",     num_h5t,   sim->p.exp_halfKd);
	my_read( , "/params/inv_exp_halfKu", num_h5t,   sim->p.inv_exp_halfKu);
	my_read( , "/params/inv_exp_halfKd", num_h5t,   sim->p.inv_exp_halfKd);
	my_read(_double, "/params/exp_lambda",     sim->p.exp_lambda);
	my_read(_double, "/params/del",            sim->p.del);
	my_read(_double, "/params/D",              sim->php.D);
	my_read(_int, "/params/map_munu",            sim->php.map_munu);
	my_read(_int, "/params/D_nums_nonzero",      sim->php.D_nums_nonzero);
	my_read(_int, "/params/D_nonzero_inds",      sim->php.D_nonzero_inds);
	my_read(_double, "/params/local_box_widths", sim->php.local_box_widths);
	my_read(_double, "/params/block_box_widths", sim->php.block_box_widths);
	my_read(_double, "/params/ph_masses", sim->php.masses);
	my_read(_double, "/params/gmat",           sim->php.gmat);
	my_read(_int, "/params/num_coupled_to_X",      sim->php.num_coupled_to_X);
	my_read(_int, "/params/ind_coupled_to_X",      sim->php.ind_coupled_to_X);
	my_read(_int,    "/params/F",             &sim->p.F);
	my_read(_int,    "/params/n_sweep",       &sim->p.n_sweep);
	my_read( ,       "/state/rng", H5T_NATIVE_UINT64, sim->s.rng);
	my_read(_int,    "/state/sweep",          &sim->s.sweep);
	my_read(_int,    "/state/hs",              sim->s.hs);
	my_read(_double, "/state/X",               sim->s.X);
	my_read(_double, "/state/exp_X",           sim->s.exp_X);
	my_read(_double, "/state/inv_exp_X",       sim->s.inv_exp_X);
	my_read(_int,    "/meas_eqlt/n_sample",   &sim->m_eq.n_sample);
	my_read( , "/meas_eqlt/sign",        num_h5t, &sim->m_eq.sign);
	my_read( , "/meas_eqlt/density",     num_h5t, sim->m_eq.density);
	my_read( , "/meas_eqlt/double_occ",  num_h5t, sim->m_eq.double_occ);
	my_read( , "/meas_eqlt/g00",         num_h5t, sim->m_eq.g00);
	my_read( , "/meas_eqlt/nn",          num_h5t, sim->m_eq.nn);
	my_read( , "/meas_eqlt/xx",          num_h5t, sim->m_eq.xx);
	my_read( , "/meas_eqlt/zz",          num_h5t, sim->m_eq.zz);
	my_read( , "/meas_eqlt/pair_sw",     num_h5t, sim->m_eq.pair_sw);
	if (sim->p.meas_energy_corr) {
		my_read( , "/meas_eqlt/kk", num_h5t, sim->m_eq.kk);
		my_read( , "/meas_eqlt/kv", num_h5t, sim->m_eq.kv);
		my_read( , "/meas_eqlt/kn", num_h5t, sim->m_eq.kn);
		my_read( , "/meas_eqlt/vv", num_h5t, sim->m_eq.vv);
		my_read( , "/meas_eqlt/vn", num_h5t, sim->m_eq.vn);
	}
	if (sim->p.period_uneqlt > 0) {
		my_read(_int,    "/meas_uneqlt/n_sample", &sim->m_ue.n_sample);
		my_read( , "/meas_uneqlt/sign",      num_h5t, &sim->m_ue.sign);
		my_read( , "/meas_uneqlt/gt0",       num_h5t, sim->m_ue.gt0);
		my_read( , "/meas_uneqlt/nn",        num_h5t, sim->m_ue.nn);
		my_read( , "/meas_uneqlt/xx",        num_h5t, sim->m_ue.xx);
		my_read( , "/meas_uneqlt/zz",        num_h5t, sim->m_ue.zz);
		my_read( , "/meas_uneqlt/pair_sw",   num_h5t, sim->m_ue.pair_sw);
		if (sim->p.meas_bond_corr) {
			my_read( , "/meas_uneqlt/pair_bb", num_h5t, sim->m_ue.pair_bb);
			my_read( , "/meas_uneqlt/jj",      num_h5t, sim->m_ue.jj);
			my_read( , "/meas_uneqlt/jsjs",    num_h5t, sim->m_ue.jsjs)
			my_read( , "/meas_uneqlt/kk",      num_h5t, sim->m_ue.kk);
			my_read( , "/meas_uneqlt/ksks",    num_h5t, sim->m_ue.ksks);
		}
		if (sim->p.meas_energy_corr) {
			my_read( , "/meas_uneqlt/kv", num_h5t, sim->m_ue.kv);
			my_read( , "/meas_uneqlt/kn", num_h5t, sim->m_ue.kn);
			my_read( , "/meas_uneqlt/vv", num_h5t, sim->m_ue.vv);
			my_read( , "/meas_uneqlt/vn", num_h5t, sim->m_ue.vn);
		}
		if (sim->p.meas_nematic_corr) {
			my_read( , "/meas_uneqlt/nem_nnnn", num_h5t, sim->m_ue.nem_nnnn);
			my_read( , "/meas_uneqlt/nem_ssss", num_h5t, sim->m_ue.nem_ssss);
		}
	}
	my_read(_int, "/meas_ph/n_sample",       &sim->m_ph.n_sample);
	my_read(_int, "/meas_ph/n_local_accept", sim->m_ph.n_local_accept);
	my_read(_int, "/meas_ph/n_local_total",  sim->m_ph.n_local_total);
	my_read(_int, "/meas_ph/n_block_accept", sim->m_ph.n_block_accept);
	my_read(_int, "/meas_ph/n_block_total",  sim->m_ph.n_block_total);
	my_read(_int, "/meas_ph/n_flip_accept", sim->m_ph.n_flip_accept);
	my_read(_int, "/meas_ph/n_flip_total",  sim->m_ph.n_flip_total);
	my_read( , "/meas_ph/sign",        num_h5t, &sim->m_ph.sign);

	my_read( , "/meas_ph/X_avg",    num_h5t, sim->m_ph.X_avg);
	my_read( , "/meas_ph/X_avg_sq", num_h5t, sim->m_ph.X_avg_sq);
	my_read( , "/meas_ph/X_sq_avg", num_h5t, sim->m_ph.X_sq_avg);
	my_read( , "/meas_ph/X_cubed_avg", num_h5t, sim->m_ph.X_cubed_avg);
	my_read( , "/meas_ph/V_avg",    num_h5t, sim->m_ph.V_avg);
	my_read( , "/meas_ph/V_sq_avg", num_h5t, sim->m_ph.V_sq_avg);
	my_read( , "/meas_ph/PE",       num_h5t, sim->m_ph.PE);
	my_read( , "/meas_ph/KE",       num_h5t, sim->m_ph.KE);
	my_read( , "/meas_ph/nX",       num_h5t, sim->m_ph.nX);
	my_read( , "/meas_ph/XX",       num_h5t, sim->m_ph.XX);

#undef my_read

	status = H5Fclose(file_id);
	return_if(status < 0, -1, "H5Fclose() failed: %d\n", status);
	return 0;
}

int sim_data_save(const struct sim_data *sim)
{
	const hid_t file_id = H5Fopen(sim->file, H5F_ACC_RDWR, H5P_DEFAULT);
	return_if(file_id < 0, -1, "H5Fopen() failed: %ld\n", file_id);

	herr_t status;
	hid_t dset_id;

#define my_write(name, type, data) do { \
	dset_id = H5Dopen2(file_id, (name), H5P_DEFAULT); \
	return_if(dset_id < 0, -1, "H5Dopen2() failed for %s: %ld\n", name, dset_id); \
	status = H5Dwrite(dset_id, (type), H5S_ALL, H5S_ALL, H5P_DEFAULT, (data)); \
	return_if(status < 0, -1, "H5Dwrite() failed for %s: %d\n", name, status); \
	status = H5Dclose(dset_id); \
	return_if(status < 0, -1, "H5Dclose() failed for %s: %d\n", name, status); \
} while (0);

	my_write("/state/rng",            H5T_NATIVE_UINT64,  sim->s.rng);
	my_write("/state/sweep",          H5T_NATIVE_INT,    &sim->s.sweep);
	my_write("/state/hs",             H5T_NATIVE_INT,     sim->s.hs);
	my_write("/state/X",              H5T_NATIVE_DOUBLE,  sim->s.X);
	my_write("/state/exp_X",          H5T_NATIVE_DOUBLE,  sim->s.exp_X);
	my_write("/state/inv_exp_X",      H5T_NATIVE_DOUBLE,  sim->s.inv_exp_X);
	my_write("/meas_eqlt/n_sample",   H5T_NATIVE_INT,    &sim->m_eq.n_sample);
	my_write("/meas_eqlt/sign",       num_h5t, &sim->m_eq.sign);
	my_write("/meas_eqlt/density",    num_h5t,  sim->m_eq.density);
	my_write("/meas_eqlt/double_occ", num_h5t,  sim->m_eq.double_occ);
	my_write("/meas_eqlt/g00",        num_h5t,  sim->m_eq.g00);
	my_write("/meas_eqlt/nn",         num_h5t,  sim->m_eq.nn);
	my_write("/meas_eqlt/xx",         num_h5t,  sim->m_eq.xx);
	my_write("/meas_eqlt/zz",         num_h5t,  sim->m_eq.zz);
	my_write("/meas_eqlt/pair_sw",    num_h5t,  sim->m_eq.pair_sw);
	if (sim->p.meas_energy_corr) {
		my_write("/meas_eqlt/kk", num_h5t, sim->m_eq.kk);
		my_write("/meas_eqlt/kv", num_h5t, sim->m_eq.kv);
		my_write("/meas_eqlt/kn", num_h5t, sim->m_eq.kn);
		my_write("/meas_eqlt/vv", num_h5t, sim->m_eq.vv);
		my_write("/meas_eqlt/vn", num_h5t, sim->m_eq.vn);
	}
	if (sim->p.period_uneqlt > 0) {
		my_write("/meas_uneqlt/n_sample", H5T_NATIVE_INT,    &sim->m_ue.n_sample);
		my_write("/meas_uneqlt/sign",     num_h5t, &sim->m_ue.sign);
		my_write("/meas_uneqlt/gt0",      num_h5t,  sim->m_ue.gt0);
		my_write("/meas_uneqlt/nn",       num_h5t,  sim->m_ue.nn);
		my_write("/meas_uneqlt/xx",       num_h5t,  sim->m_ue.xx);
		my_write("/meas_uneqlt/zz",       num_h5t,  sim->m_ue.zz);
		my_write("/meas_uneqlt/pair_sw",  num_h5t,  sim->m_ue.pair_sw);
		if (sim->p.meas_bond_corr) {
			my_write("/meas_uneqlt/pair_bb", num_h5t, sim->m_ue.pair_bb);
			my_write("/meas_uneqlt/jj",      num_h5t, sim->m_ue.jj);
			my_write("/meas_uneqlt/jsjs",    num_h5t, sim->m_ue.jsjs);
			my_write("/meas_uneqlt/kk",      num_h5t, sim->m_ue.kk);
			my_write("/meas_uneqlt/ksks",    num_h5t, sim->m_ue.ksks);
		}
		if (sim->p.meas_energy_corr) {
			my_write("/meas_uneqlt/kv", num_h5t, sim->m_ue.kv);
			my_write("/meas_uneqlt/kn", num_h5t, sim->m_ue.kn);
			my_write("/meas_uneqlt/vv", num_h5t, sim->m_ue.vv);
			my_write("/meas_uneqlt/vn", num_h5t, sim->m_ue.vn);
		}
		if (sim->p.meas_nematic_corr) {
			my_write("/meas_uneqlt/nem_nnnn", num_h5t, sim->m_ue.nem_nnnn);
			my_write("/meas_uneqlt/nem_ssss", num_h5t, sim->m_ue.nem_ssss);
		}
	}
	my_write("/meas_ph/n_sample",       H5T_NATIVE_INT, &sim->m_ph.n_sample);
	my_write("/meas_ph/n_local_accept", H5T_NATIVE_INT, sim->m_ph.n_local_accept);
	my_write("/meas_ph/n_local_total",  H5T_NATIVE_INT, sim->m_ph.n_local_total);
	my_write("/meas_ph/n_block_accept", H5T_NATIVE_INT, sim->m_ph.n_block_accept);
	my_write("/meas_ph/n_block_total",  H5T_NATIVE_INT, sim->m_ph.n_block_total);
	my_write("/meas_ph/n_flip_accept", H5T_NATIVE_INT, sim->m_ph.n_flip_accept);
	my_write("/meas_ph/n_flip_total",  H5T_NATIVE_INT, sim->m_ph.n_flip_total);
	my_write("/meas_ph/sign",  num_h5t, &sim->m_ph.sign);
	my_write("/meas_ph/X_avg",    num_h5t, sim->m_ph.X_avg);
	my_write("/meas_ph/X_avg_sq", num_h5t, sim->m_ph.X_avg_sq);
	my_write("/meas_ph/X_sq_avg", num_h5t, sim->m_ph.X_sq_avg);
	my_write("/meas_ph/X_cubed_avg", num_h5t, sim->m_ph.X_cubed_avg);
	my_write("/meas_ph/V_avg",    num_h5t, sim->m_ph.V_avg);
	my_write("/meas_ph/V_sq_avg", num_h5t, sim->m_ph.V_sq_avg);
	my_write("/meas_ph/PE",       num_h5t, sim->m_ph.PE);
	my_write("/meas_ph/KE",       num_h5t, sim->m_ph.KE);
	my_write("/meas_ph/nX",       num_h5t, sim->m_ph.nX);
	my_write("/meas_ph/XX",       num_h5t, sim->m_ph.XX);

#undef my_write

	status = H5Fclose(file_id);
	return_if(status < 0, -1, "H5Fclose() failed: %d\n", status);
	return 0;
}

void sim_data_free(const struct sim_data *sim)
{
	if (sim->p.period_uneqlt > 0) {
		if (sim->p.meas_nematic_corr) {
			my_free(sim->m_ue.nem_ssss);
			my_free(sim->m_ue.nem_nnnn);
		}
		if (sim->p.meas_energy_corr) {
			my_free(sim->m_ue.vn);
			my_free(sim->m_ue.vv);
			my_free(sim->m_ue.kn);
			my_free(sim->m_ue.kv);
		}
		if (sim->p.meas_bond_corr) {
			my_free(sim->m_ue.ksks);
			my_free(sim->m_ue.kk);
			my_free(sim->m_ue.jsjs);
			my_free(sim->m_ue.jj);
			my_free(sim->m_ue.pair_bb);
		}
		my_free(sim->m_ue.pair_sw);
		my_free(sim->m_ue.zz);
		my_free(sim->m_ue.xx);
		my_free(sim->m_ue.nn);
		my_free(sim->m_ue.gt0);
	}
	if (sim->p.meas_energy_corr) {
		my_free(sim->m_eq.vn);
		my_free(sim->m_eq.vv);
		my_free(sim->m_eq.kn);
		my_free(sim->m_eq.kv);
		my_free(sim->m_eq.kk);
	}
	my_free(sim->m_ph.n_local_accept);
	my_free(sim->m_ph.n_local_total);
	my_free(sim->m_ph.n_block_accept);
	my_free(sim->m_ph.n_block_total);
	my_free(sim->m_ph.n_flip_accept);
	my_free(sim->m_ph.n_flip_total);
	my_free(sim->m_ph.X_avg);
	my_free(sim->m_ph.X_avg_sq);
	my_free(sim->m_ph.X_sq_avg);
	my_free(sim->m_ph.X_cubed_avg);
	my_free(sim->m_ph.V_avg);
	my_free(sim->m_ph.V_sq_avg);
	my_free(sim->m_ph.PE);
	my_free(sim->m_ph.KE);
	my_free(sim->m_ph.nX);
	my_free(sim->m_ph.XX);
	my_free(sim->m_eq.pair_sw);
	my_free(sim->m_eq.zz);
	my_free(sim->m_eq.xx);
	my_free(sim->m_eq.nn);
	my_free(sim->m_eq.g00);
	my_free(sim->m_eq.double_occ);
	my_free(sim->m_eq.density);
	my_free(sim->s.hs);
	my_free(sim->s.X);
	my_free(sim->s.exp_X);
	my_free(sim->s.inv_exp_X);
	my_free(sim->php.gmat);
	my_free(sim->php.D);
	my_free(sim->php.map_munu);
	my_free(sim->php.D_nums_nonzero);
	my_free(sim->php.D_nonzero_inds);
	my_free(sim->php.local_box_widths);
	my_free(sim->php.block_box_widths);
	my_free(sim->php.masses);
	my_free(sim->php.num_coupled_to_X);
	my_free(sim->php.ind_coupled_to_X);
	my_free(sim->p.del);
	my_free(sim->p.exp_lambda);
	my_free(sim->p.inv_exp_halfKd);
	my_free(sim->p.inv_exp_halfKu);
	my_free(sim->p.exp_halfKd);
	my_free(sim->p.exp_halfKu);
	my_free(sim->p.inv_exp_Kd);
	my_free(sim->p.inv_exp_Ku);
	my_free(sim->p.exp_Kd);
	my_free(sim->p.exp_Ku);
	my_free(sim->p.degen_bb);
	my_free(sim->p.degen_bs);
	my_free(sim->p.degen_ij);
	my_free(sim->p.degen_i);
//	my_free(sim->p.U);
//	my_free(sim->p.K);
	my_free(sim->p.peierlsd);
	my_free(sim->p.peierlsu);
	my_free(sim->p.map_bb);
	my_free(sim->p.map_bs);
	my_free(sim->p.bonds);
	my_free(sim->p.map_ij);
	my_free(sim->p.map_i);
}
