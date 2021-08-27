"""Run the whole pipeline (pre-processing, CDL and plot figures) for a bunch
of hyper-parameters configuration."""

# import itertools
# from joblib import Parallel, delayed

# from camcan_process_to_evoked_parallel import MEG_preproc
# from run_CSC_grad import run_csc
# from plot_CSC_grad import plot_csc
# %%

from utils import run_csc, plot_csc


# === some subjects info ===
# CC110037 - age: 18.75
# CC620264 - age: 76.33 (default subject in Tim's)
# CC723395 - age: 86.08

# Experience parameters

exp_params = {'subject_id': 'CC620264',
              'sfreq': 150.,  # in Tim's: 300
              'atom_duration': 0.7,  # in Tim's: 0.5,
              'n_atoms': 30,  # in Tim's: 25,
              'reg': 0.2,  # in Tim's: 0.2,
              'eps': 1e-5,  # in Tim's: 1e-4,
              'tol_z': 1e-3  # in Tim's: 1e-2
              }
# Default: only a Maxwell filter is applied and a GreedyCDL is run on full data
# ie, apply_maxwell_filter = True; apply_ica_cleaning = False;
# use_greedy_cdl = True; cdl_on_epoch = False

# res = run_csc(**exp_params)
plot_csc(exp_params, activeStartTime=1.7, shift_acti=False)


# # Parameters to vary
# list_n_atoms = [30]
# list_reg = [0.1]

# # number of job for parallel computing
# n_jobs = 5


# ========== Define gloabl pipeline ==========


# def procedure(comb):
#     print('Run for comb %s' % comb)
#     # unpack hyper-parameters values
#     n_atoms, reg = comb
#     # run CSC
#     run_csc(subjectID=subjectID, cdl_on_epoch=cdl_on_epoch, n_atoms=n_atoms,
#             atomDuration=atomDuration, sfreq=sfreq,
#             use_batch_cdl=use_batch_cdl, use_greedy_cdl=use_greedy_cdl,
#             reg=reg, eps=eps, tol_z=tol_z,
#             subtract_first_samp=subtract_first_samp,
#             use_drago=use_drago)
#     # plot and save results
#     plot_csc(subjectID=subjectID, cdl_on_epoch=cdl_on_epoch, n_atoms=n_atoms,
#              atomDuration=atomDuration, sfreq=sfreq,
#              use_batch_cdl=use_batch_cdl, use_greedy_cdl=use_greedy_cdl,
#              reg=reg, eps=eps, tol_z=tol_z,
#              activeStartTime=activeStartTime, shift_acti=shift_acti,
#              use_drago=use_drago)

#     return None

# # ========== Parallel pipeline ==========


# if __name__ == '__main__':
#     # # run preprocess for the subject ID
#     # print("Run pre-processing")
#     # MEG_preproc(subjectID=subjectID,
#     # apply_maxwell_filter=apply_maxwell_filter)

#     # # define hyper-parameters combinations
#     # combs = itertools.product(list_n_atoms, list_reg)
#     # # Run pipeline in parallel
#     # print("Run parallel")
#     # res = Parallel(n_jobs=n_jobs, verbose=1)(delayed(procedure)(this_comb)
#     #                                          for this_comb in combs)

#     print("Run pre-processing")
#     MEG_preproc(subjectID=subjectID,
#                 apply_maxwell_filter=apply_maxwell_filter,
#                 use_drago=use_drago)

#     print("Run CSC")
#     run_csc(subjectID=subjectID, cdl_on_epoch=cdl_on_epoch, n_atoms=n_atoms,
#             atomDuration=atomDuration, sfreq=sfreq,
#             use_batch_cdl=use_batch_cdl, use_greedy_cdl=use_greedy_cdl,
#             reg=reg, eps=eps, tol_z=tol_z, use_drago=use_drago)

#     print("Plot figures")
#     plot_csc(subjectID=subjectID, cdl_on_epoch=cdl_on_epoch, n_atoms=n_atoms,
#              atomDuration=atomDuration, sfreq=sfreq,
#              use_batch_cdl=use_batch_cdl, use_greedy_cdl=use_greedy_cdl,
#              reg=reg, eps=eps, tol_z=tol_z,
#              activeStartTime=activeStartTime, shift_acti=shift_acti,
#              use_drago=use_drago)

# %%
