import os
import shutil
import sys

import h5py
import numpy as np
from scipy.linalg import expm

np.seterr(over="ignore")
np.set_printoptions(precision=3)

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

## 
##  Consider four unit cells a, b, c, d
## 
##   c    d
##   |\  /
##   | \/
##   | /\
##   |/  \
##   a----b
## 
##  A bond_hoppings_t struct describes the minimum bonds between these unit cells such that
##  repeating the bonds everywhere on a rectangular lattice with periodic
##  boundary conditions tessellates all the bonds in the lattice.
## 
##  Each t.xx element in the struct is an (Norb * Norb) length array.
##  For instance, if t.ab[0 + 2 * Norb] == 3.456, that means between orbital 0 of
##  unit cell a and orbital 2 of unit cell b, the hopping parameter is 3.456.
## 
def create_1(file_sim=None, file_params=None, overwrite=False, init_rng=None,
             Nx=4, Ny=4, Norb = 3, mu=-0.3, tpd=1.13, tpp=0.49, Udd=8.5, Upp = 0.0, dpd = 3.24, dt=0.125, L=64,
             nflux=0,
             K=0.1, Kp=0.0625, Kpp=0.025, phonon_k=0.05, M_Cu=1, M_O=0.25,
             g_Cu=0, g_Ox=0.5, g_Oy=0.5,
             local_box_width_Cu=0.85, local_box_width_Ox=1.85, local_box_width_Oy=1.85,
             num_local_updates=48,
             block_box_width_Cu=2.75, block_box_width_Ox=4, block_box_width_Oy=4,
             num_block_updates=12,
             num_flip_updates=12,
             track_phonon_ite=0,
             n_delay=16, n_matmul=8, n_sweep_warm=200, n_sweep_meas=2000,
             period_eqlt=8, period_uneqlt=0,
             meas_bond_corr=1, meas_energy_corr=0, meas_nematic_corr=0,
             meas_thermal = 0, meas_2bond_corr=0,
             trans_sym=1,
             checkpoint_every=1000):
    assert L % n_matmul == 0 and L % period_eqlt == 0
    Ncell = Nx * Ny
    N = Norb * Ncell
    assert num_local_updates <= N and num_block_updates <= N and num_flip_updates <= N

    if nflux != 0:
        dtype_num = np.complex128
    else:
        dtype_num = np.float64

    if init_rng is None:
        init_rng = rand_seed_urandom()
    rng = init_rng.copy()
    init_hs = np.zeros((L, N), dtype=np.int32)
    nd = 3
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
        # map_i = np.zeros(N, dtype=np.int32)
        # degen_i = np.array((N,), dtype=np.int32)

        map_i = np.array(list(range(Norb))*Ncell, dtype = np.int32)
        degen_i = np.array((Ncell,)*Norb, dtype=np.int32)
    else:
        map_i = np.arange(N, dtype=np.int32)        
        degen_i = np.ones(N, dtype=np.int32)
    num_i = map_i.max() + 1
    assert num_i == degen_i.size

    # 2 site mapping
    map_ij = np.zeros((N, N), dtype=np.int32)
    # num_ij = N if trans_sym else N*N
    num_ij = Norb*Norb*Ncell if trans_sym else N*N
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

                    # map_ij[jx + Nx*jy, ix + Nx*iy] = k
                    # degen_ij[k] += 1

                    for p in range(Norb):
                        for o in range(Norb):
                            map_ij[p + Norb*(jx + Nx*jy), o + Norb*(ix + Nx*iy)] = o + Norb*p + Norb*Norb*k
                            degen_ij[o + Norb*p + Norb*Norb*k] += 1

    assert num_ij == map_ij.max() + 1

    # bond definitions
    bps = 2 + 4 + 4 # bonds per cell
    num_b = bps*Ncell # total bonds in cluster
    bonds = np.zeros((2, num_b), dtype = np.int32)
    for iy in range(Ny):
        for ix in range(Nx):
            i = ix + Nx*iy
            ipy = (iy + 1) % Ny
            ipx = (ix + 1) % Nx
            imx = (ix - 1) % Nx
            imy = (iy - 1) % Ny

            i1 = ipx + Nx*iy
            i2 = ix  + Nx*ipy
            i3 = imx + Nx*iy
            i4 = ix  + Nx*imy
            i5 = ipx + Nx*imy

            for ibond, icell in enumerate([i1, i2], 0): # Cu - Cu
                bonds[0, i + ibond*Ncell] = i     * Norb + 0
                bonds[1, i + ibond*Ncell] = icell * Norb + 0

            for ibond, (icell, iorb) in enumerate([(i , 1), (i , 2), (i3, 1), (i4, 2)], 2): # Cu - O
                bonds[0, i + ibond*Ncell] = i     * Norb + 0
                bonds[1, i + ibond*Ncell] = icell * Norb + iorb

            for ibond, icell in enumerate([i1, i, i4, i5], 6): # O - O
                bonds[0, i + ibond*Ncell] = i     * Norb + 1
                bonds[1, i + ibond*Ncell] = icell * Norb + 2

    # bps = 4 if tp != 0.0 else 2  # bonds per site
    # num_b = bps*N  # total bonds in cluster
    # bonds = np.zeros((2, num_b), dtype=np.int32)
    # for iy in range(Ny):
    #     for ix in range(Nx):
    #         i = ix + Nx*iy
    #         iy1 = (iy + 1) % Ny
    #         ix1 = (ix + 1) % Nx
    #         bonds[0, i] = i            # i0 = i
    #         bonds[1, i] = ix1 + Nx*iy  # i1 = i + x
    #         bonds[0, i + N] = i            # i0 = i
    #         bonds[1, i + N] = ix + Nx*iy1  # i1 = i + y
    #         if bps == 4:
    #             bonds[0, i + 2*N] = i             # i0 = i
    #             bonds[1, i + 2*N] = ix1 + Nx*iy1  # i1 = i + x + y
    #             bonds[0, i + 3*N] = ix1 + Nx*iy   # i0 = i + x
    #             bonds[1, i + 3*N] = ix + Nx*iy1   # i1 = i + y

    # 1 bond 1 site mapping
    map_bs = np.zeros((N, num_b), dtype=np.int32)
    num_bs = bps*N if trans_sym else num_b*N
    degen_bs = np.zeros(num_bs, dtype=np.int32)
    for j in range(N):
        # for i in range(N):
        for i in range(Ncell):            
            # k = map_ij[j, i]
            k = map_ij[j, i*Norb]//Norb
            for ib in range(bps):
                # kk = k + num_ij*ib
                kk = k + num_ij//Norb*ib
                # map_bs[j, i + N*ib] = kk
                map_bs[j, i + Ncell*ib] = kk # v
                degen_bs[kk] += 1
    #assert num_bs == map_bs.max() + 1
    assert num_bs == map_bs.max() + 1

    # 2 bond mapping
    map_bb = np.zeros((num_b, num_b), dtype=np.int32)
    # num_bb = bps*bps*N if trans_sym else num_b*num_b
    num_bb = bps*bps*Ncell if trans_sym else num_b*num_b
    degen_bb = np.zeros(num_bb, dtype = np.int32)
    for j in range(Ncell):
        for i in range(Ncell):
            k = map_ij[j*Norb, i*Norb]//Norb//Norb
            for jb in range(bps):
                for ib in range(bps):
                    # kk = k + num_ij*(ib + bps*jb)
                    kk = k + num_ij//Norb//Norb*(ib + bps*jb)
                    # map_bb[j + N*jb, i + N*ib] = kk
                    map_bb[j + Ncell*jb, i + Ncell*ib] = kk
                    degen_bb[kk] += 1
    assert num_bb == map_bb.max() + 1

#if meas_2bond_corr == 0:
    b2ps = 0
    num_b2 = b2ps*Ncell
    bond2s = np.zeros((2, num_b2), dtype=np.int32)
    
    num_b2b2 = b2ps*b2ps*Ncell if trans_sym else num_b2*num_b2
    map_b2b2 = np.zeros((num_b2, num_b2), dtype=np.int32)
    degen_b2b2 = np.zeros(num_b2b2, dtype = np.int32)
    
    num_bb2 = bps*b2ps*N if trans_sym else num_b*num_b2
    map_bb2 = np.zeros((num_b, num_b2), dtype=np.int32)
    degen_bb2 = np.zeros(num_bb2, dtype = np.int32)
    
    num_b2b = b2ps*bps*N if trans_sym else num_b2*num_b
    map_b2b = np.zeros((num_b2, num_b), dtype=np.int32)
    degen_b2b = np.zeros(num_b2b, dtype = np.int32)

## cell indices
##  c    d
##  |\  /
##  | \/
##  | /\
##  |/  \
##  a----b

    taa = np.zeros((Norb, Norb))
    tab = np.zeros((Norb, Norb))
    tac = np.zeros((Norb, Norb))
    tad = np.zeros((Norb, Norb))
    tbc = np.zeros((Norb, Norb))
    taa[0, 1] =  tpd
    taa[0, 2] = -tpd
    taa[1, 2] = -tpp
    tab[1, 0] = -tpd
    tab[1, 2] =  tpp
    tac[2, 0] =  tpd
    tac[2, 1] =  tpp
    tbc[2, 1] = -tpp

    # hopping (periodic boundaries and no field)
    tij = np.zeros((N, N), dtype=np.float64) # why?
    for iy in range(Ny):
        for ix in range(Nx):
            for p in range(Norb):
                for o in range(Norb):
                    px = (ix + 1)%Nx
                    py = (iy + 1)%Ny
                    a = ix + Nx*iy
                    b = px + Nx*iy
                    c = ix + Nx*py
                    d = px + Nx*py

                    tij[a*Norb+p, a*Norb+o] += taa[p, o]
                    tij[a*Norb+o, a*Norb+p] += taa[p, o]

                    tij[a*Norb+p, b*Norb+o] += tab[p, o]
                    tij[b*Norb+o, a*Norb+p] += tab[p, o]

                    tij[a*Norb+p, c*Norb+o] += tac[p, o]
                    tij[c*Norb+o, a*Norb+p] += tac[p, o]

                    tij[b*Norb+p, c*Norb+o] += tbc[p, o]
                    tij[c*Norb+o, b*Norb+p] += tbc[p, o]

    # alpha = 0.5  # gauge choice. 0.5 for symmetric gauge.
    # beta = 1 - alpha
    # phi = np.zeros((Ny*Nx, Ny*Nx))
    # # path is straight line
    # # if Ny is even, prefer dy - -Ny/2 over Ny/2. likewise for even Nx
    # for dy in range((1-Ny)//2, (1+Ny)//2):
    #     for dx in range((1-Nx)//2, (1+Nx)//2):
    #         for iy in range(Ny):
    #             for ix in range(Nx):
    #                 jy = iy + dy
    #                 jjy = jy % Ny
    #                 offset_y = jy - jjy
    #                 jx = ix + dx
    #                 jjx = jx % Nx
    #                 offset_x = jx - jjx
    #                 mx = (ix + jx)/2
    #                 my = (iy + jy)/2
    #                 phi[jjx + Nx*jjy, ix + Nx*iy] = \
    #                     -alpha*my*dx + beta*mx*dy - beta*offset_x*jy + alpha*offset_y*jx - alpha*offset_x*offset_y
    # peierls = np.exp(2j*np.pi*(nflux/(Ny*Nx))*phi)

    peierls = np.ones((N, N))
    thermal_phases = np.ones((b2ps, N))
    Ku = -tij.real

    # if dtype_num == np.complex128:
    #     Ku = -tij * peierls
    #     assert np.max(np.abs(Ku - Ku.T.conj())) < 1e-10
    # else:
    #     Ku = -tij.real
    #     assert np.max(np.abs(peierls.imag)) < 1e-10
    #     peierls = peierls.real
    #     assert np.max(np.abs(thermal_phases.imag)) < 1e-10
    #     thermal_phases = thermal_phases.real


    for i in range(Ny*Nx):
        Ku[Norb*i,   Norb*i  ] += -mu+Udd/2 
        Ku[Norb*i+1, Norb*i+1] += -mu+Upp/2 + dpd
        Ku[Norb*i+2, Norb*i+2] += -mu+Upp/2 + dpd

    exp_Ku = expm(-dt * Ku)
    inv_exp_Ku = expm(dt * Ku)
    exp_halfKu = expm(-dt/2 * Ku)
    inv_exp_halfKu = expm(dt/2 * Ku)
#   exp_K = np.array(mpm.expm(mpm.matrix(-dt * K)).tolist(), dtype=np.float64)

    if trans_sym:
        U_i = np.array([Udd, Upp, Upp], dtype=np.float64)
    else:
        U_i = np.array([Udd, Upp, Upp]*Ncell, dtype=np.float64)

    # U_i = U*np.ones_like(degen_i, dtype=np.float64)
    assert U_i.shape[0] == num_i

    exp_lmbd = np.exp(0.5*U_i*dt) + np.sqrt(np.expm1(U_i*dt))
#    exp_lmbd = np.exp(np.arccosh(np.exp(0.5*U_i*dt)))
#    exp_lmbd = float(mpm.exp(mpm.acosh(mpm.exp(0.5*float(U*dt)))))
    exp_lambda = np.array((exp_lmbd[map_i]**-1, exp_lmbd[map_i]))
    delll = np.array((exp_lmbd[map_i]**2 - 1, exp_lmbd[map_i]**-2 - 1))

    if trans_sym:
        local_box_widths = np.array([local_box_width_Cu,
                                     local_box_width_Ox,
                                     local_box_width_Oy], dtype=np.float64)
        block_box_widths = np.array([block_box_width_Cu,
                                     block_box_width_Ox,
                                     block_box_width_Oy], dtype=np.float64)
        ph_masses = np.array([M_Cu, M_O, M_O], dtype=np.float64)
    else:
        local_box_widths = np.array([local_box_width_Cu,
                                     local_box_width_Ox,
                                     local_box_width_Oy] * Ncell, dtype=np.float64)
        block_box_widths = np.array([block_box_width_Cu,
                                     block_box_width_Ox,
                                     block_box_width_Oy] * Ncell, dtype=np.float64)
        ph_masses = np.array([M_Cu, M_O, M_O] * Ncell, dtype=np.float64)

    div = np.uint64(1) << np.uint64(53)
    for l in range(L):
        for m in range(nd):
            for i in range(Ncell):
                for o in range(Norb):
                    uniform_rand = (rand_uint(rng) >> np.uint64(11)) / div
                    io = o + i * Norb
                    init_X[l, m, io] = (uniform_rand - 0.5) * \
                        local_box_widths[map_i[io]]

    # map_munu[mu, nu] = munu (like the munu index in D)
    # For example, map_munu[2, 1] = 4 means that zy interaction is the 4th
    # place in the D matrix
    map_munu = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2]], dtype=np.int32)

    # Phonon D-matrix: D_munu(R, R') where mu, nu = x, y or z
    # The generic matrix looks like [xx, yy, zz, xy, yz, zx] where each
    # subblock is an N x N matrix D(R, R') = D(|R - R'|). The meaning
    # of D is different from A&M eq. 22.46, here D is like a look-up
    # table, i.e. D_munu(R, R') gives the interaction between
    # X_mu(R) and X_nu(R'). This is useful when proposing change in MC.
    # For example, X_mu(R) -> X_mu(R) + dX, then use D to look up all
    # the Harmonic terms DX_mu(R)X_nu(R') in the Hamiltonian that are
    # affected by this change.
    num_munu = 6
    D = np.zeros((num_munu, N, N))
    for i in range(Ncell):
        iy, ix = i // Nx, i % Nx
        iyp1, iym1 = (iy + 1) % Ny, (iy - 1) % Ny
        ixp1, ixm1 = (ix + 1) % Nx, (ix - 1) % Nx
        imx = ixm1 + Nx * iy
        imy = ix + Nx * iym1
        ipx = ixp1 + Nx * iy
        ipy = ix + Nx * iyp1
        for o in range(Norb):
            io = o + i*Norb
            if o == 0:  # Cu
                D[0, io, io] = 2 * (K + Kp)
                D[0, io, 1+i*Norb] = -K
                D[0, io, 1+imx*Norb] = -K
                D[0, io, 2+i*Norb] = -Kp
                D[0, io, 2+imy*Norb] = -Kp
                D[1, io, io] = 2 * (K + Kp)
                D[1, io, 1+i*Norb] = -Kp
                D[1, io, 1+imx*Norb] = -Kp
                D[1, io, 2+i*Norb] = -K
                D[1, io, 2+imy*Norb] = -K
                D[2, io, io] = 4 * Kpp
                D[2, io, 1+i*Norb] = -Kpp
                D[2, io, 1+imx*Norb] = -Kpp
                D[2, io, 2+i*Norb] = -Kpp
                D[2, io, 2+imy*Norb] = -Kpp
            elif o == 1:  # Ox
                D[0, io, io] = 2 * K
                D[0, io, i*Norb] = -K
                D[0, io, ipx*Norb] = -K
                D[1, io, io] = 2 * Kp
                D[1, io, i*Norb] = -Kp
                D[1, io, ipx*Norb] = -Kp
                D[2, io, io] = 2 * Kpp
                D[2, io, i*Norb] = -Kpp
                D[2, io, ipx*Norb] = -Kpp
            else:  # Oy
                D[0, io, io] = 2 * Kp
                D[0, io, i*Norb] = -Kp
                D[0, io, ipy*Norb] = -Kp
                D[1, io, io] = 2 * K
                D[1, io, i*Norb] = -K
                D[1, io, ipy*Norb] = -K
                D[2, io, io] = 2 * Kpp
                D[2, io, i*Norb] = -Kpp
                D[2, io, ipy*Norb] = -Kpp

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
    for i in range(Ncell):
        gmat[2,   i*Norb,   i*Norb] = g_Cu
        gmat[2, 1+i*Norb, 1+i*Norb] = g_Ox
        gmat[2, 2+i*Norb, 2+i*Norb] = g_Oy

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
        f["metadata"]["Norb"] = Norb
        f["metadata"]["bps"] = bps
        f["metadata"]["b2ps"] = b2ps
        f["metadata"]["Udd"] = Udd
        f["metadata"]["Upp"] = Upp
        f["metadata"]["tpd"] = tpd
        f["metadata"]["tpp"] = tpp
        f["metadata"]["dpd"] = dpd
        f["metadata"]["nflux"] = nflux
        f["metadata"]["mu"] = mu
        f["metadata"]["beta"] = L*dt
        f["metadata"]["K"] = K
        f["metadata"]["Kp"] = Kp
        f["metadata"]["Kpp"] = Kpp
        f["metadata"]["M_Cu"] = M_Cu
        f["metadata"]["M_O"] = M_O
        f["metadata"]["g_Cu"] = g_Cu
        f["metadata"]["g_Ox"] = g_Ox
        f["metadata"]["g_Oy"] = g_Oy

        # parameters used by dqmc code
        f.create_group("params")
        # model parameters
        f["params"]["N"] = np.array(N, dtype=np.int32)
        f["params"]["L"] = np.array(L, dtype=np.int32)
        f["params"]["map_i"] = map_i
        f["params"]["map_ij"] = map_ij
        f["params"]["bonds"] = bonds
        f["params"]["bond2s"] = bond2s
        f["params"]["map_bs"] = map_bs
        f["params"]["map_bb"] = map_bb
        f["params"]["map_b2b"] = map_b2b
        f["params"]["map_bb2"] = map_bb2
        f["params"]["map_b2b2"] = map_b2b2
        f["params"]["peierlsu"] = peierls
        f["params"]["peierlsd"] = f["params"]["peierlsu"]
        f["params"]["pp_u"] = thermal_phases.conj()
        f["params"]["pp_d"] = thermal_phases.conj()
        f["params"]["ppr_u"] = thermal_phases
        f["params"]["ppr_d"] = thermal_phases
        f["params"]["Ku"] = Ku
        f["params"]["Kd"] = f["params"]["Ku"]
        f["params"]["U"] = U_i
        f["params"]["dt"] = np.array(dt, dtype=np.float64)
        f["params"]["inv_dt_sq"] = np.array(1 / dt ** 2, dtype=np.float64)
        f["params"]["chem_pot"] = np.array(mu, dtype=np.float64)
        f["params"]["nd"] = np.array(nd, dtype=np.int32)
        f["params"]["num_munu"] = np.array(num_munu, dtype=np.int32)
        f["params"]["map_munu"] = map_munu
        f["params"]["D"] = D
        f["params"]["D_nums_nonzero"] = D_nums_nonzero
        f["params"]["max_D_nums_nonzero"] = np.array(max_D_nums_nonzero, dtype=np.int32)
        f["params"]["D_nonzero_inds"] = D_nonzero_inds
        f["params"]["phonon_k"] = np.array(phonon_k, dtype=np.float64)
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
        f["params"]["meas_thermal"] = meas_thermal
        f["params"]["meas_2bond_corr"] = meas_2bond_corr
        f["params"]["meas_energy_corr"] = meas_energy_corr
        f["params"]["meas_nematic_corr"] = meas_nematic_corr
        f["params"]["checkpoint_every"] = checkpoint_every

        # precalculated stuff
        f["params"]["num_i"] = num_i
        f["params"]["num_ij"] = num_ij
        f["params"]["num_b"] = num_b
        f["params"]["num_b2"] = num_b2
        f["params"]["num_bs"] = num_bs
        f["params"]["num_bb"] = num_bb
        f["params"]["num_b2b"] = num_b2b
        f["params"]["num_bb2"] = num_bb2
        f["params"]["num_b2b2"] = num_b2b2
        f["params"]["degen_i"] = degen_i
        f["params"]["degen_ij"] = degen_ij
        f["params"]["degen_bs"] = degen_bs
        f["params"]["degen_bb"] = degen_bb
        f["params"]["degen_bb2"] = degen_bb2
        f["params"]["degen_b2b"] = degen_b2b
        f["params"]["degen_b2b2"] = degen_b2b2
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
        f["state"]["partial_write"] = 0

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
        f["meas_ph"]["X_cubed_avg"] = np.zeros(num_i*nd, dtype=dtype_num)
        f["meas_ph"]["V_avg"] = np.zeros(num_i*nd, dtype=dtype_num)
        f["meas_ph"]["V_sq_avg"] = np.zeros(num_i*nd, dtype=dtype_num)
        f["meas_ph"]["PE"] = np.zeros(num_i*nd, dtype=dtype_num)
        f["meas_ph"]["KE"] = np.zeros(num_i*nd, dtype=dtype_num)
        f["meas_ph"]["nX"] = np.zeros(num_ij*nd*L, dtype=dtype_num)
        f["meas_ph"]["XX"] = np.zeros(num_ij*nd*nd*L, dtype=dtype_num)

def create_batch(Nfiles=1, prefix=None, seed=None, Nstart=0, **kwargs):
    if seed is None:
        init_rng = rand_seed_urandom()
    else:
        init_rng = rand_seed_splitmix64(seed)

    if prefix is None:
        prefix = "sim"

    file_0 = "{}_{}.h5".format(prefix, Nstart)
    file_p = "{}.h5.params".format(prefix)

    create_1(file_sim=file_0, file_params=file_p, init_rng=init_rng, **kwargs)

    with h5py.File(file_p, "r") as f:
        N = f["params"]["N"][...]
        L = f["params"]["L"][...]
        nd = f["params"]["nd"][...]
        Norb = f["metadata"]["Norb"][...]
        local_box_widths = f["params"]["local_box_widths"][...]
        map_i = f["params"]["map_i"][...]
        gmat = f["params"]["gmat"][...]
        dt = f["params"]["dt"][...]
        Ncell = N // Norb

    for i in range(Nstart+1, Nfiles):
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
                for r in range(Ncell):
                    for o in range(Norb):
                        uniform_rand = (rand_uint(rng) >> np.uint64(11)) / div
                        io = o + r * Norb
                        init_X[l, m, io] = (uniform_rand - 0.5) * \
                            local_box_widths[map_i[io]]
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
          file_0 if Nfiles == Nstart+1 else "{} ... {}".format(file_0, file_i))
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
