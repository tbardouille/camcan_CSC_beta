"""Run the whole pipeline (pre-processing, CDL and plot figures) for a bunch
of hyper-parameters configuration."""

import itertools
from joblib import Parallel, delayed

from camcan_process_to_evoked_parallel import MEG_preproc
from run_CSC_grad import run_csc
from plot_CSC_grad import plot_csc


# ========== Global parameters ==========
# Fixed hyper-parameters
use_drago = True
subjectID = 'CC620264'
apply_maxwell_filter = True  # in Tim's: False (already done on input data)
cdl_on_epoch = False  # in Tim's: True
n_atoms = 30  # in Tim's: 25
atomDuration = 0.7  # in Tim's: 0.5
sfreq = 150.  # in Tim's: 300
use_batch_cdl = False  # in Tim's: True
use_greedy_cdl = True  # in Tim's: False
reg = 0.3  # in Tim's: 0.2
eps = 1e-5  # in Tim's: 1e-4
tol_z = 1e-3  # in Tim's: 1e-2
activeStartTime = 1.7  # in Tim's: 1.7
shift_acti = True  # in Tim's: False

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
            reg=reg, eps=eps, tol_z=tol_z, use_drago=use_drago)
    # plot and save results
    plot_csc(subjectID=subjectID, cdl_on_epoch=cdl_on_epoch, n_atoms=n_atoms,
             atomDuration=atomDuration, sfreq=sfreq,
             use_batch_cdl=use_batch_cdl, use_greedy_cdl=use_greedy_cdl,
             reg=reg, eps=eps, tol_z=tol_z,
             activeStartTime=activeStartTime, shift_acti=shift_acti,
             use_drago=use_drago)

    return None

# ========== Parallel pipeline ==========


if __name__ == '__main__':
    # # run preprocess for the subject ID
    # print("Run pre-processing")
    # MEG_preproc(subjectID=subjectID,
    # apply_maxwell_filter=apply_maxwell_filter)

    # # define hyper-parameters combinations
    # combs = itertools.product(list_n_atoms, list_reg)
    # # Run pipeline in parallel
    # print("Run parallel")
    # res = Parallel(n_jobs=n_jobs, verbose=1)(delayed(procedure)(this_comb)
    #                                          for this_comb in combs)

    print("Run pre-processing")
    MEG_preproc(subjectID=subjectID,
                apply_maxwell_filter=apply_maxwell_filter,
                use_drago=use_drago)

    print("Run CSC")
    run_csc(subjectID=subjectID, cdl_on_epoch=cdl_on_epoch, n_atoms=n_atoms,
            atomDuration=atomDuration, sfreq=sfreq,
            use_batch_cdl=use_batch_cdl, use_greedy_cdl=use_greedy_cdl,
            reg=reg, eps=eps, tol_z=tol_z, use_drago=use_drago)

    print("Plot figures")
    plot_csc(subjectID=subjectID, cdl_on_epoch=cdl_on_epoch, n_atoms=n_atoms,
             atomDuration=atomDuration, sfreq=sfreq,
             use_batch_cdl=use_batch_cdl, use_greedy_cdl=use_greedy_cdl,
             reg=reg, eps=eps, tol_z=tol_z,
             activeStartTime=activeStartTime, shift_acti=shift_acti,
             use_drago=use_drago)
