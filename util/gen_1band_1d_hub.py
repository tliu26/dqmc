import os
import shutil
import sys

import h5py
import numpy as np
from scipy.linalg import expm

np.seterr(over="ignore")


def rand_seed_urandom():
    rng = np.zeros(17, dtype=np.uint64)
    rng[:16] = np.frombuffer(os.urandom(16*8), dtype=np.uint64)
    return rng

# http://xoroshiro.di.unimi.it/splitmix64.c
def rand_seed_splitmix64(x):
    x = np.uint64(x)
    rng = np.zeros(17, dtype=np.uint64)
    for i in range(16):
        x += np.uint64(0x9E3779B97F4A7C15)
        z = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        rng[i] = z ^ (z >> np.uint64(31))
    return rng


# http://xoroshiro.di.unimi.it/xorshift1024star.c
def rand_uint(rng):
    s0 = rng[rng[16]]
    p = (int(rng[16]) + 1) & 15
    rng[16] = p
    s1 = rng[p]
    s1 ^= s1 << np.uint64(31)
    rng[p] = s1 ^ s0 ^ (s1 >> np.uint64(11)) ^ (s0 >> np.uint64(30))
    return rng[p] * np.uint64(1181783497276652981)


def rand_jump(rng):
    JMP = np.array((0x84242f96eca9c41d,
                    0xa3c65b8776f96855, 0x5b34a39f070b5837, 0x4489affce4f31a1e,
                    0x2ffeeb0a48316f40, 0xdc2d9891fe68c022, 0x3659132bb12fea70,
                    0xaac17d8efa43cab8, 0xc4cb815590989b13, 0x5ee975283d71c93b,
                    0x691548c86c1bd540, 0x7910c41d10a1e6a5, 0x0b5fc64563b3e2a8,
                    0x047f7684e9fc949d, 0xb99181f2d8f685ca, 0x284600e3f30e38c3
                    ), dtype=np.uint64)

    t = np.zeros(16, dtype=np.uint64)
    for i in range(16):
        for b in range(64):
            if JMP[i] & (np.uint64(1) << np.uint64(b)):
                for j in range(16):
                    t[j] ^= rng[(np.uint64(j) + rng[16]) & np.uint64(15)]
            rand_uint(rng)

    for j in range(16):
        rng[(np.uint64(j) + rng[16]) & np.uint64(15)] = t[j]


def create_1(file_sim=None, file_params=None, overwrite=False, init_rng=None,
             Nx=16, Ny=4, mu=0.0, tp=0.0, U=6.0, dt=0.115, L=40,
             nflux=0,
             omega=0.2, J=0, g0=0.05, g1=0.01, g2=0,
             local_box_width=2, num_local_updates=64,
             block_box_width=22, num_block_updates=16,
             num_flip_updates=16,
             track_phonon_ite=0,
             n_delay=16, n_matmul=8, n_sweep_warm=200, n_sweep_meas=2000,
             period_eqlt=8, period_uneqlt=0,
             meas_bond_corr=1, meas_energy_corr=0, meas_nematic_corr=0,
             trans_sym=1):
    assert L % n_matmul == 0 and L % period_eqlt == 0
    N = Nx * Ny
    assert num_local_updates <= N and num_block_updates <= N and num_flip_updates <= N

    if nflux != 0:
        dtype_num = np.complex128
    else:
        dtype_num = np.float64

    if init_rng is None:
        init_rng = rand_seed_urandom()
    rng = init_rng.copy()
    init_hs = np.zeros((L, N), dtype=np.int32)
    nd = 1
    init_X = np.zeros((L, nd, N), dtype=np.float64)

    if file_sim is None:
        file_sim = "sim.h5"
    if file_params is None:
        file_params = file_sim
    
    one_file = (os.path.abspath(file_sim) == os.path.abspath(file_params))

    for l in range(L):
        for i in range(N):
            init_hs[l, i] = rand_uint(rng) >> np.uint64(63)

    # 1 site mapping
    if trans_sym:
        map_i = np.zeros(N, dtype=np.int32)
        degen_i = np.array((N,), dtype=np.int32)
    else:
        map_i = np.arange(N, dtype=np.int32)
        degen_i = np.ones(N, dtype=np.int32)
    num_i = map_i.max() + 1
    assert num_i == degen_i.size

    # 2 site mapping
    map_ij = np.zeros((N, N), dtype=np.int32)
    num_ij = N if trans_sym else N*N
    degen_ij = np.zeros(num_ij, dtype=np.int32)
    for jy in range(Ny):
        for jx in range(Nx):
            for iy in range(Ny):
                for ix in range(Nx):
                    if trans_sym:
                        ky = (iy - jy) % Ny
                        kx = (ix - jx) % Nx
                        k = kx + Nx*ky
                    else:
                        k = (ix + Nx*iy) + N*(jx + Nx*jy)
                    map_ij[jx + Nx*jy, ix + Nx*iy] = k
                    degen_ij[k] += 1
    assert num_ij == map_ij.max() + 1

    # bond definitions
    bps = 4 if tp != 0.0 else 2  # bonds per site
    num_b = bps*N  # total bonds in cluster
    bonds = np.zeros((2, num_b), dtype=np.int32)
    for iy in range(Ny):
        for ix in range(Nx):
            i = ix + Nx*iy
            iy1 = (iy + 1) % Ny
            ix1 = (ix + 1) % Nx
            bonds[0, i] = i            # i0 = i
            bonds[1, i] = ix1 + Nx*iy  # i1 = i + x
            bonds[0, i + N] = i            # i0 = i
            bonds[1, i + N] = ix + Nx*iy1  # i1 = i + y
            if bps == 4:
                bonds[0, i + 2*N] = i             # i0 = i
                bonds[1, i + 2*N] = ix1 + Nx*iy1  # i1 = i + x + y
                bonds[0, i + 3*N] = ix1 + Nx*iy   # i0 = i + x
                bonds[1, i + 3*N] = ix + Nx*iy1   # i1 = i + y

    # 1 bond 1 site mapping
    map_bs = np.zeros((N, num_b), dtype=np.int32)
    num_bs = bps*N if trans_sym else num_b*N
    degen_bs = np.zeros(num_bs, dtype=np.int32)
    for j in range(N):
        for i in range(N):
            k = map_ij[j, i]
            for ib in range(bps):
                kk = k + num_ij*ib
                map_bs[j, i + N*ib] = kk
                degen_bs[kk] += 1
    assert num_bs == map_bs.max() + 1

    # 2 bond mapping
    map_bb = np.zeros((num_b, num_b), dtype=np.int32)
    num_bb = bps*bps*N if trans_sym else num_b*num_b
    degen_bb = np.zeros(num_bb, dtype = np.int32)
    for j in range(N):
        for i in range(N):
            k = map_ij[j, i]
            for jb in range(bps):
                for ib in range(bps):
                    kk = k + num_ij*(ib + bps*jb)
                    map_bb[j + N*jb, i + N*ib] = kk
                    degen_bb[kk] += 1
    assert num_bb == map_bb.max() + 1

    # hopping (assuming periodic boundaries and no field)
    tij = np.zeros((Ny*Nx, Ny*Nx), dtype=np.complex128)
    for iy in range(Ny):
        for ix in range(Nx):
            iy1 = (iy + 1) % Ny
            ix1 = (ix + 1) % Nx
                #jx    jy    ix    iy
            tij[ix + Nx*iy1, ix + Nx*iy] += 1
            tij[ix + Nx*iy, ix + Nx*iy1] += 1
            tij[ix1 + Nx*iy, ix + Nx*iy] += 1
            tij[ix + Nx*iy, ix1 + Nx*iy] += 1

            tij[ix1 + Nx*iy1, ix + Nx*iy] += tp
            tij[ix + Nx*iy, ix1 + Nx*iy1] += tp
            tij[ix1 + Nx*iy, ix + Nx*iy1] += tp
            tij[ix + Nx*iy1, ix1 + Nx*iy] += tp

    alpha = 0.5  # gauge choice. 0.5 for symmetric gauge.
    beta = 1 - alpha
    phi = np.zeros((Ny*Nx, Ny*Nx))
    # path is straight line
    # if Ny is even, prefer dy - -Ny/2 over Ny/2. likewise for even Nx
    for dy in range((1-Ny)//2, (1+Ny)//2):
        for dx in range((1-Nx)//2, (1+Nx)//2):
            for iy in range(Ny):
                for ix in range(Nx):
                    jy = iy + dy
                    jjy = jy % Ny
                    offset_y = jy - jjy
                    jx = ix + dx
                    jjx = jx % Nx
                    offset_x = jx - jjx
                    mx = (ix + jx)/2
                    my = (iy + jy)/2
                    phi[jjx + Nx*jjy, ix + Nx*iy] = \
                        -alpha*my*dx + beta*mx*dy - beta*offset_x*jy + alpha*offset_y*jx - alpha*offset_x*offset_y
    peierls = np.exp(2j*np.pi*(nflux/(Ny*Nx))*phi)

    if dtype_num == np.complex128:
        Ku = -tij * peierls
        assert np.max(np.abs(Ku - Ku.T.conj())) < 1e-10
    else:
        Ku = -tij.real
        assert np.max(np.abs(peierls.imag)) < 1e-10
        peierls = peierls.real

    for i in range(Ny*Nx):
        Ku[i, i] -= mu

    exp_Ku = expm(-dt * Ku)
    inv_exp_Ku = expm(dt * Ku)
    exp_halfKu = expm(-dt/2 * Ku)
    inv_exp_halfKu = expm(dt/2 * Ku)
#   exp_K = np.array(mpm.expm(mpm.matrix(-dt * K)).tolist(), dtype=np.float64)

    U_i = U*np.ones_like(degen_i, dtype=np.float64)
    assert U_i.shape[0] == num_i

    exp_lmbd = np.exp(0.5*U_i*dt) + np.sqrt(np.expm1(U_i*dt))
#    exp_lmbd = np.exp(np.arccosh(np.exp(0.5*U_i*dt)))
#    exp_lmbd = float(mpm.exp(mpm.acosh(mpm.exp(0.5*float(U*dt)))))
    exp_lambda = np.array((exp_lmbd[map_i]**-1, exp_lmbd[map_i]))
    delll = np.array((exp_lmbd[map_i]**2 - 1, exp_lmbd[map_i]**-2 - 1))

    local_box_widths = np.array([local_box_width] * num_i, dtype=np.float64)
    block_box_widths = np.array([block_box_width] * num_i, dtype=np.float64)
    ph_masses = np.ones(num_i, dtype=np.float64);

    div = np.uint64(1) << np.uint64(53)
    for l in range(L):
        for m in range(nd):
            for i in range(N):
                uniform_rand = (rand_uint(rng) >> np.uint64(11)) / div
                init_X[l, m, i] = (uniform_rand - 0.5) * local_box_widths[map_i[i]]

    # map_munu[mu, nu] = munu (like the munu index in D)
    # For example, map_munu[2, 1] = 4 means that zy interaction is the 4th
    # place in the D matrix
    map_munu = np.array([[0]], dtype=np.int32)

    # Phonon D-matrix: D_munu(R, R') where mu, nu = x, y or z
    # The generic matrix looks like [xx, yy, zz, xy, yz, zx] where each
    # subblock is an N x N matrix D(R, R') = D(|R - R'|). The meaning
    # of D is different from A&M eq. 22.46, here D is like a look-up
    # table, i.e. D_munu(R, R') gives the interaction between
    # X_mu(R) and X_nu(R'). This is useful when proposing change in MC.
    # For example, X_mu(R) -> X_mu(R) + dX, then use D to look up all
    # the Harmonic terms DX_mu(R)X_nu(R') in the Hamiltonian that are
    # affected by this change.
    num_munu = 1
    D = np.zeros((num_munu, N, N))
    omega_sq = 0.5 * omega ** 2 + 2 * J
    for i in range(N):
        D[0, i, i] = omega_sq
        ix, iy = i % Nx, i // Nx
        ipx = (ix + 1) % Nx + iy * Nx
        imx = (ix - 1) % Nx + iy * Nx
        ipy = ix + ((iy + 1) % Ny) * Nx
        imy = ix + ((iy - 1) % Ny) * Nx
        D[0, i, ipx] = -J/2
        D[0, i, imx] = -J/2
        D[0, i, ipy] = -J/2
        D[0, i, imy] = -J/2

    # D_nums_nonzero = [[xx, yy, zz, xy, yz, zx] for i = 0,
    #                   [xx, yy, zz, xy, yz, zx] for i = 1,
    #                   ...
    #                  ]
    # D_nums_nonzero.shape = [N, num_munu]
    # Number of nonzero D matrix elements. For example,
    # D_num_nonzero[i, 0] = num x oscillators coupled to u_ix
    # D_num_nonzero[i, 3] = num y oscillators coupled to u_ix, etc.
    D_ = np.swapaxes(D, 0, 1)
    D_nums_nonzero = np.sum(D_ != 0, axis=-1, dtype=np.int32)
    max_D_nums_nonzero = D_nums_nonzero.max()

    # D_nonzero_inds = [[xj0, xj1, ...], [yj0, yj1, ...], ... for i = 0,
    #                   [xj0, xj1, ...], [yj0, yj1, ...], ... for i = 1,
    #                   ...
    #                  ]
    # D_nonzero_inds.shape = [N, num_munu, max(D_nums_nonzero)]
    # Indices of sites that couple to i.
    # D_nonzero_inds[i, 0, 0] = j such that u_jx and u_ix are coupled,
    # D_nonzero_inds[i, 0, 1] = another j such that u_jx and u_ix are coupled,
    # D_nonzero_inds[i, 1, 0] = j such that u_jy and u_iy are coupled, etc.
    D_nonzero_inds = np.zeros((N, num_munu, max_D_nums_nonzero), dtype=np.int32)
    for i in range(N):
        for munu in range(num_munu):
            jj = 0
            for j, Dij in enumerate(D_[i, munu]):
                if Dij != 0:
                    D_nonzero_inds[i, munu, jj] = j
                    jj += 1

    # Holstein electron-phonon interaction
    # H_ep = \sum_sigma \sum_{m,i,j} g_{m,i,j} X_{m,i} n_{j,\sigma} where
    # g_{m,i,j} = gmat[m, i, j]
    gmat = np.zeros((nd, N, N), dtype=np.float64)
    m = 0
    for iy in range(Ny):
        for ix in range(Nx):
            gmat[m, ix + Nx*iy, ix + Nx*iy] += g0
            iyp1 = (iy + 1) % Ny
            ixp1 = (ix + 1) % Nx
            iym1 = (iy - 1) % Ny
            ixm1 = (ix - 1) % Nx
            gmat[m, ix + Nx*iy, ix + Nx*iyp1] += g1
            gmat[m, ix + Nx*iy, ix + Nx*iym1] += g1
            gmat[m, ix + Nx*iy, ixp1 + Nx*iy] += g1
            gmat[m, ix + Nx*iy, ixm1 + Nx*iy] += g1
            gmat[m, ix + Nx*iy, ixp1 + Nx*iyp1] += g2
            gmat[m, ix + Nx*iy, ixm1 + Nx*iyp1] += g2
            gmat[m, ix + Nx*iy, ixp1 + Nx*iym1] += g2
            gmat[m, ix + Nx*iy, ixm1 + Nx*iym1] += g2

    gmat_mask = (gmat != 0).any(0)
    # num_coupled_to_X[i] = number of sites on which electrons (n_j) are coupled to
    # X_{m,i} for any m
    num_coupled_to_X = np.sum(gmat_mask, axis=-1)
    # ind_coupled_to_X[i, jj] = index j of the jjth electron that are coupled to
    # X_{m,i} for any m
    ind_coupled_to_X = np.zeros((N, num_coupled_to_X.max()), dtype=np.int32) - 1
    for i in range(N):
        jj = 0
        for j in range(N):
            if gmat_mask[i, j]:
                ind_coupled_to_X[i, jj] = j
                jj += 1

    # hep.shape = ((L, nd, N, 1) * ( , nd, N, N)).sum(1, 2)
    #           = (L, nd, N, N).sum(1, 2)
    #           = (L, N)
    hep = np.sum(init_X[..., None] * gmat, axis=(1, 2))
    init_exp_X = np.exp(-dt*hep)
    init_inv_exp_X = np.exp(dt*hep)

    with h5py.File(file_params, "w" if overwrite else "x") as f:
        # parameters not used by dqmc code, but useful for analysis
        f.create_group("metadata")
        f["metadata"]["version"] = 0.1
        f["metadata"]["model"] = \
            "Hubbard (complex)" if dtype_num == np.complex128 else "Hubbard"
        f["metadata"]["Nx"] = Nx
        f["metadata"]["Ny"] = Ny
        f["metadata"]["bps"] = bps
        f["metadata"]["U"] = U
        f["metadata"]["t'"] = tp
        f["metadata"]["nflux"] = nflux
        f["metadata"]["mu"] = mu
        f["metadata"]["beta"] = L*dt
        f["metadata"]["omega"] = omega
        f["metadata"]["J"] = J
        f["metadata"]["g0"] = g0
        f["metadata"]["g1"] = g1
        f["metadata"]["g2"] = g2

        # parameters used by dqmc code
        f.create_group("params")
        # model parameters
        f["params"]["N"] = np.array(N, dtype=np.int32)
        f["params"]["L"] = np.array(L, dtype=np.int32)
        f["params"]["map_i"] = map_i
        f["params"]["map_ij"] = map_ij
        f["params"]["bonds"] = bonds
        f["params"]["map_bs"] = map_bs
        f["params"]["map_bb"] = map_bb
        f["params"]["peierlsu"] = peierls
        f["params"]["peierlsd"] = f["params"]["peierlsu"]
        f["params"]["Ku"] = Ku
        f["params"]["Kd"] = f["params"]["Ku"]
        f["params"]["U"] = U_i
        f["params"]["dt"] = np.array(dt, dtype=np.float64)
        f["params"]["inv_dt_sq"] = np.array(1 / dt ** 2, dtype=np.float64)
        f["params"]["chem_pot"] = np.array(mu, dtype=np.float64)
        f["params"]["nd"] = nd
        f["params"]["num_munu"] = num_munu
        f["params"]["map_munu"] = map_munu
        f["params"]["D"] = D
        f["params"]["D_nums_nonzero"] = D_nums_nonzero
        f["params"]["max_D_nums_nonzero"] = np.array(max_D_nums_nonzero, dtype=np.int32)
        f["params"]["D_nonzero_inds"] = D_nonzero_inds
        f["params"]["gmat"] = gmat
        f["params"]["max_num_coupled_to_X"] = np.array(num_coupled_to_X.max(), dtype=np.int32)
        f["params"]["num_coupled_to_X"] = num_coupled_to_X
        f["params"]["ind_coupled_to_X"] = ind_coupled_to_X
        f["params"]["local_box_widths"] = local_box_widths
        f["params"]["num_local_updates"] = np.array(num_local_updates, dtype=np.int32)
        f["params"]["block_box_widths"] = block_box_widths
        f["params"]["num_block_updates"] = np.array(num_block_updates, dtype=np.int32)
        f["params"]["num_flip_updates"] = np.array(num_flip_updates, dtype=np.int32)
        f["params"]["ph_masses"] = ph_masses
        f["params"]["track_phonon_ite"] = np.array(track_phonon_ite, dtype=np.int32)

        # simulation parameters
        f["params"]["n_matmul"] = np.array(n_matmul, dtype=np.int32)
        f["params"]["n_delay"] = np.array(n_delay, dtype=np.int32)
        f["params"]["n_sweep_warm"] = np.array(n_sweep_warm, dtype=np.int32)
        f["params"]["n_sweep_meas"] = np.array(n_sweep_meas, dtype=np.int32)
        f["params"]["period_eqlt"] = np.array(period_eqlt, dtype=np.int32)
        f["params"]["period_uneqlt"] = np.array(period_uneqlt, dtype=np.int32)
        f["params"]["meas_bond_corr"] = meas_bond_corr
        f["params"]["meas_energy_corr"] = meas_energy_corr
        f["params"]["meas_nematic_corr"] = meas_nematic_corr

        # precalculated stuff
        f["params"]["num_i"] = num_i
        f["params"]["num_ij"] = num_ij
        f["params"]["num_b"] = num_b
        f["params"]["num_bs"] = num_bs
        f["params"]["num_bb"] = num_bb
        f["params"]["degen_i"] = degen_i
        f["params"]["degen_ij"] = degen_ij
        f["params"]["degen_bs"] = degen_bs
        f["params"]["degen_bb"] = degen_bb
        f["params"]["exp_Ku"] = exp_Ku
        f["params"]["exp_Kd"] = f["params"]["exp_Ku"]
        f["params"]["inv_exp_Ku"] = inv_exp_Ku
        f["params"]["inv_exp_Kd"] = f["params"]["inv_exp_Ku"]
        f["params"]["exp_halfKu"] = exp_halfKu
        f["params"]["exp_halfKd"] = f["params"]["exp_halfKu"]
        f["params"]["inv_exp_halfKu"] = inv_exp_halfKu
        f["params"]["inv_exp_halfKd"] = f["params"]["inv_exp_halfKu"]
        f["params"]["exp_lambda"] = exp_lambda
        f["params"]["del"] = delll
        f["params"]["F"] = np.array(L//n_matmul, dtype=np.int32)
        f["params"]["n_sweep"] = np.array(n_sweep_warm + n_sweep_meas,
                                          dtype=np.int32)

    with h5py.File(file_sim, "a" if one_file else "w" if overwrite else "x") as f:
        # simulation state
        params_relpath = os.path.relpath(file_params, os.path.dirname(file_sim))
        f["params_file"] = params_relpath
        if not one_file:
            f["metadata"] = h5py.ExternalLink(params_relpath, "metadata")
            f["params"] = h5py.ExternalLink(params_relpath, "params")

        f.create_group("state")
        f["state"]["sweep"] = np.array(0, dtype=np.int32)
        f["state"]["init_rng"] = init_rng  # save if need to replicate data
        f["state"]["rng"] = rng
        f["state"]["hs"] = init_hs
        f["state"]["X"] = init_X
        f["state"]["exp_X"] = init_exp_X
        f["state"]["inv_exp_X"] = init_inv_exp_X

        # measurements
        f.create_group("meas_eqlt")
        f["meas_eqlt"]["n_sample"] = np.array(0, dtype=np.int32)
        f["meas_eqlt"]["sign"] = np.array(0.0, dtype=dtype_num)
        f["meas_eqlt"]["density"] = np.zeros(num_i, dtype=dtype_num)
        f["meas_eqlt"]["double_occ"] = np.zeros(num_i, dtype=dtype_num)
        f["meas_eqlt"]["g00"] = np.zeros(num_ij, dtype=dtype_num)
        f["meas_eqlt"]["nn"] = np.zeros(num_ij, dtype=dtype_num)
        f["meas_eqlt"]["xx"] = np.zeros(num_ij, dtype=dtype_num)
        f["meas_eqlt"]["zz"] = np.zeros(num_ij, dtype=dtype_num)
        f["meas_eqlt"]["pair_sw"] = np.zeros(num_ij, dtype=dtype_num)
        if meas_energy_corr:
            f["meas_eqlt"]["kk"] = np.zeros(num_bb, dtype=dtype_num)
            f["meas_eqlt"]["kv"] = np.zeros(num_bs, dtype=dtype_num)
            f["meas_eqlt"]["kn"] = np.zeros(num_bs, dtype=dtype_num)
            f["meas_eqlt"]["vv"] = np.zeros(num_ij, dtype=dtype_num)
            f["meas_eqlt"]["vn"] = np.zeros(num_ij, dtype=dtype_num)

        if period_uneqlt > 0:
            f.create_group("meas_uneqlt")
            f["meas_uneqlt"]["n_sample"] = np.array(0, dtype=np.int32)
            f["meas_uneqlt"]["sign"] = np.array(0.0, dtype=dtype_num)
            f["meas_uneqlt"]["gt0"] = np.zeros(num_ij*L, dtype=dtype_num)
            f["meas_uneqlt"]["nn"] = np.zeros(num_ij*L, dtype=dtype_num)
            f["meas_uneqlt"]["xx"] = np.zeros(num_ij*L, dtype=dtype_num)
            f["meas_uneqlt"]["zz"] = np.zeros(num_ij*L, dtype=dtype_num)
            f["meas_uneqlt"]["pair_sw"] = np.zeros(num_ij*L, dtype=dtype_num)
            if meas_bond_corr:
                f["meas_uneqlt"]["pair_bb"] = np.zeros(num_bb*L, dtype=dtype_num)
                f["meas_uneqlt"]["jj"] = np.zeros(num_bb*L, dtype=dtype_num)
                f["meas_uneqlt"]["jsjs"] = np.zeros(num_bb*L, dtype=dtype_num)
                f["meas_uneqlt"]["kk"] = np.zeros(num_bb*L, dtype=dtype_num)
                f["meas_uneqlt"]["ksks"] = np.zeros(num_bb*L, dtype=dtype_num)
            if meas_energy_corr:
                f["meas_uneqlt"]["kv"] = np.zeros(num_bs*L, dtype=dtype_num)
                f["meas_uneqlt"]["kn"] = np.zeros(num_bs*L, dtype=dtype_num)
                f["meas_uneqlt"]["vv"] = np.zeros(num_ij*L, dtype=dtype_num)
                f["meas_uneqlt"]["vn"] = np.zeros(num_ij*L, dtype=dtype_num)
            if meas_nematic_corr:
                f["meas_uneqlt"]["nem_nnnn"] = np.zeros(num_bb*L, dtype=dtype_num)
                f["meas_uneqlt"]["nem_ssss"] = np.zeros(num_bb*L, dtype=dtype_num)
        f.create_group("meas_ph")
        f["meas_ph"]["n_sample"] = np.array(0, dtype=np.int32)
        f["meas_ph"]["n_local_accept"] = np.zeros(num_i, dtype=np.int32)
        f["meas_ph"]["n_local_total"] = np.zeros(num_i, dtype=np.int32)
        f["meas_ph"]["n_block_accept"] = np.zeros(num_i, dtype=np.int32)
        f["meas_ph"]["n_block_total"] = np.zeros(num_i, dtype=np.int32)
        f["meas_ph"]["n_flip_accept"] = np.zeros(num_i, dtype=np.int32)
        f["meas_ph"]["n_flip_total"] = np.zeros(num_i, dtype=np.int32)
        f["meas_ph"]["sign"] = np.array(0.0, dtype=dtype_num)

        f["meas_ph"]["X_avg"] = np.zeros(num_i*nd, dtype=dtype_num)
        f["meas_ph"]["X_avg_sq"] = np.zeros(num_i*nd, dtype=dtype_num)
        f["meas_ph"]["X_sq_avg"] = np.zeros(num_i*nd, dtype=dtype_num)
        f["meas_ph"]["V_avg"] = np.zeros(num_i*nd, dtype=dtype_num)
        f["meas_ph"]["V_sq_avg"] = np.zeros(num_i*nd, dtype=dtype_num)
        f["meas_ph"]["PE"] = np.zeros(num_i*nd, dtype=dtype_num)
        f["meas_ph"]["KE"] = np.zeros(num_i*nd, dtype=dtype_num)
        f["meas_ph"]["nX"] = np.zeros(num_ij*nd*L, dtype=dtype_num)
        f["meas_ph"]["XX"] = np.zeros(num_ij*nd*nd*L, dtype=dtype_num)


def create_batch(Nfiles=1, prefix=None, seed=None, **kwargs):
    if seed is None:
        init_rng = rand_seed_urandom()
    else:
        init_rng = rand_seed_splitmix64(seed)

    if prefix is None:
        prefix = "sim"

    file_0 = "{}_{}.h5".format(prefix, 0)
    file_p = "{}.h5.params".format(prefix)

    create_1(file_sim=file_0, file_params=file_p, init_rng=init_rng, **kwargs)

    with h5py.File(file_p, "r") as f:
        N = f["params"]["N"][...]
        L = f["params"]["L"][...]
        nd = f["params"]["nd"][...]
        local_box_widths = f["params"]["local_box_widths"][...]
        map_i = f["params"]["map_i"][...]
        gmat = f["params"]["gmat"][...]
        dt = f["params"]["dt"][...]

    for i in range(1, Nfiles):
        rand_jump(init_rng)
        rng = init_rng.copy()
        init_hs = np.zeros((L, N), dtype=np.int32)
        init_X = np.zeros((L, nd, N), dtype=np.float64)

        for l in range(L):
            for r in range(N):
                init_hs[l, r] = rand_uint(rng) >> np.uint64(63)

        div = np.uint64(1) << np.uint64(53)
        for l in range(L):
            for m in range(nd):
                for j in range(N):
                    uniform_rand = (rand_uint(rng) >> np.uint64(11)) / div
                    init_X[l, m, j] = (uniform_rand - 0.5) * local_box_widths[map_i[j]]
        hep = np.sum(init_X[..., None] * gmat, axis=(1, 2))
        init_exp_X = np.exp(-dt*hep)
        init_inv_exp_X = np.exp(dt*hep)

        file_i = "{}_{}.h5".format(prefix, i)
        shutil.copy2(file_0, file_i)
        with h5py.File(file_i, "r+") as f:
            f["state"]["init_rng"][...] = init_rng
            f["state"]["rng"][...] = rng
            f["state"]["hs"][...] = init_hs
            f["state"]["X"][...] = init_X
            f["state"]["exp_X"][...] = init_exp_X
            f["state"]["inv_exp_X"][...] = init_inv_exp_X
    print("created simulation files:",
          file_0 if Nfiles == 1 else "{} ... {}".format(file_0, file_i))
    print("parameter file:", file_p)


def main(argv):
    kwargs = {}
    for arg in argv[1:]:
        eq = arg.find("=")
        if eq == -1:
            print("couldn't find \"=\" in argument " + arg)
            return
        key = arg[:eq]
        val = arg[(eq + 1):]
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except:
                pass
        kwargs[key] = val
    create_batch(**kwargs)

if __name__ == "__main__":
    main(sys.argv)
