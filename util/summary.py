import sys
import util
from glob import glob

def info(path):
    n_sample, sign, density, double_occ, sweep, n_sweep = \
        util.load_file(path, "meas_eqlt/n_sample", "meas_eqlt/sign",
                             "meas_eqlt/density", "meas_eqlt/double_occ",
                             "state/sweep", "params/n_sweep")
    print(f"n_sample={n_sample}, sweep={sweep}/{n_sweep}")
    if n_sample > 0:
        print(f"<sign>={(sign/n_sample)}")
        print(f"<n>={(density/sign)}")
        print(f"<m_z^2>={((density-2*double_occ)/sign)}")
    
    n_sample, n_local_accept, n_local_total, n_block_accept, n_block_total = \
        util.load_file(path, "meas_ph/n_sample",
                       "meas_ph/n_local_accept", "meas_ph/n_local_total",
                       "meas_ph/n_block_accept", "meas_ph/n_block_total")
    X_avg, X_sq_avg = util.load_file(path, "meas_ph/X_avg", "meas_ph/X_sq_avg")
    print(f"Phonon n_sample = {n_sample}")
    print(f"Phonon local acceptance = {n_local_accept} / {n_local_total}" + 
          f" = {n_local_accept/n_local_total}")
    print(f"Phonon block acceptance = {n_block_accept} / {n_block_total}" +
          f" = {n_block_accept/n_block_total}")
    n_flip_accept, n_flip_total = \
        util.load_file(path,
                       "meas_ph/n_flip_accept", "meas_ph/n_flip_total")
    print(f"Phonon flip acceptance = {n_flip_accept} / {n_flip_total}" +
          f" = {n_flip_accept/n_flip_total}")
    if n_sample > 0:
        print(f"<X>={X_avg/n_sample}")
        print(f"<X^2>={X_sq_avg/n_sample}")

def main(argv):
    #rework this function to make sure it works on Windows
    for path_spec in argv[1:]:
        files = sorted(glob(path_spec))
        if len(files) == 0:
            print("No files matching:"+path_spec)
        else:
            for f in files:
                print(f)
                info(f)

if __name__ == "__main__":
    main(sys.argv)
