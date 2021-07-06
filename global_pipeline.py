"""Run the whole pipeline (pre-processing, CDL and plot figures) for a bunch
of hyper-parameters configuration."""

import itertools
from joblib import Parallel, delayed

from camcan_process_to_evoked_parallel import MEG_preproc
from run_CSC_grad import run_csc
from plot_CSC_grad import plot_csc


# ========== Global parameters ==========
# Fixed hyper-parameters
subjectID = 'CC620264'
maxwell_filter = True
cdl_on_epoch = False
n_atoms = 25
atomDuration = 0.7
sfreq = 150.
use_batch_cdl = True
use_greedy_cdl = False
reg = 0.2
eps = 1e-4
tol_z = 1e-2
activeStartTime = 1.7
shift_acti = True

# # Parameters to vary
# list_n_atoms = [30]
# list_reg = [0.1]

# # number of job for parallel computing
# n_jobs = 5


# ========== Define gloabl pipeline ==========


def procedure(comb):
    print('Run for comb %s' % comb)
    # unpack hyper-parameters values
    n_atoms, reg = comb
    # run CSC
    run_csc(subjectID=subjectID, cdl_on_epoch=cdl_on_epoch, n_atoms=n_atoms,
            atomDuration=atomDuration, sfreq=sfreq,
            use_batch_cdl=use_batch_cdl, use_greedy_cdl=use_greedy_cdl,
            reg=reg, eps=eps, tol_z=tol_z)
    # plot and save results
    plot_csc(subjectID=subjectID, cdl_on_epoch=cdl_on_epoch, n_atoms=n_atoms,
             atomDuration=atomDuration, sfreq=sfreq,
             use_batch_cdl=use_batch_cdl, use_greedy_cdl=use_greedy_cdl,
             reg=reg, eps=eps, tol_z=tol_z,
             activeStartTime=activeStartTime, shift_acti=shift_acti)

    return None

# ========== Parallel pipeline ==========


if __name__ == '__main__':
    # # run preprocess for the subject ID
    # print("Run pre-processing")
    # MEG_preproc(subjectID=subjectID, maxwell_filter=maxwell_filter)

    # # define hyper-parameters combinations
    # combs = itertools.product(list_n_atoms, list_reg)
    # # Run pipeline in parallel
    # print("Run parallel")
    # res = Parallel(n_jobs=n_jobs, verbose=1)(delayed(procedure)(this_comb)
    #                                          for this_comb in combs)

    run_csc(subjectID=subjectID, cdl_on_epoch=cdl_on_epoch, n_atoms=n_atoms,
            atomDuration=atomDuration, sfreq=sfreq,
            use_batch_cdl=use_batch_cdl, use_greedy_cdl=use_greedy_cdl,
            reg=reg, eps=eps, tol_z=tol_z)

    plot_csc(subjectID=subjectID, cdl_on_epoch=cdl_on_epoch, n_atoms=n_atoms,
             atomDuration=atomDuration, sfreq=sfreq,
             use_batch_cdl=use_batch_cdl, use_greedy_cdl=use_greedy_cdl,
             reg=reg, eps=eps, tol_z=tol_z,
             activeStartTime=activeStartTime, shift_acti=shift_acti)
