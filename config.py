import numpy as np
from pathlib import Path


TEAM = 'dal'  # 'DAL' | 'parietal

if TEAM == 'parietal':
    # path to CSC results
    RESULTS_DIR = Path('./results_csc')
    # Paths for Cam-CAN dataset
    DATA_DIR = Path("/storage/store/data/")
    BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"  # Root path to raw BIDS files
    PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"

    BEM_DIR = DATA_DIR / "camcan-mne/freesurfer"
    TRANS_DIR = DATA_DIR / "camcan-mne/trans"

elif TEAM == 'DAL':
    RESULTS_DIR = Path('/media/NAS/lpower/CSC/results/')
    DATA_DIR = Path(
        "/media/WDEasyStore/timb/camcan/release05/cc700/meg/pipeline/release005/")
    BIDS_ROOT = DATA_DIR / "BIDSsep/smt/"  # Root path to raw BIDS files
    SSS_CAL_FILE = Path(
        "/home/timb/camcan/camcanMEGcalibrationFiles/sss_cal.dat")
    CT_SPARSE_FILE = Path(
        "/home/timb/camcan/camcanMEGcalibrationFiles/ct_sparse.fif")
    PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"
    HOME_DIR = Path("/media/NAS/lpower/CSC/")


def get_paths(subject_id, team=TEAM):
    if team == 'dal':
        # Files for dipole fitting
        subjectsDir = '/home/timb/camcan/subjects'
        transFif = subjectsDir + '/coreg/sub-' + subject_id + '-trans.fif'
        bemFif = subjectsDir + '/sub-' + subject_id + '/bem/sub-' + \
            subject_id + '-5120-bem-sol.fif'
        fifFile = '/transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif'
        epochFif = '/home/timb/camcan/proc_data/TaskSensorAnalysis_transdef/' + \
            subject_id + fifFile

    elif team == 'parietal':
        # XXX find the good paths for Parital drago server
        bemFif = BEM_DIR / subject_id / \
            'bem' / (subject_id + '-meg-bem.fif')
        transFif = TRANS_DIR / ('sub-' + subject_id + '-trans.fif')
        # XXX
        EPOCH_DIR = ''
        fifFile = ''
        epochFif = EPOCH_DIR + subject_id + fifFile

    return epochFif, transFif, bemFif


# hare all the global variables
EXP_PARAMS = {
    "sfreq": 150.,            # 300
    "atom_duration": 0.5,      # 0.5,
    "n_atoms": 20,             # 25,
    "reg": 0.2,                # 0.2,
    "eps": 1e-5,               # 1e-4,
    "tol_z": 1e-3,             # 1e-2
}

CDL_PARAMS = {
    'n_atoms': EXP_PARAMS['n_atoms'],
    'n_times_atom': int(np.round(EXP_PARAMS["atom_duration"] * EXP_PARAMS['sfreq'])),
    'rank1': True, 'uv_constraint': 'separate',
    'window': True,  # in Tim's: False
    'unbiased_z_hat': True,  # in Tim's: False
    'D_init': 'chunk',
    'lmbd_max': 'scaled', 'reg': EXP_PARAMS['reg'],
    'n_iter': 100, 'eps': EXP_PARAMS['eps'],
    'solver_z': 'lgcd',
    'solver_z_kwargs': {'tol': EXP_PARAMS['tol_z'], 'max_iter': 1000},
    'solver_d': 'alternate_adaptive',
    'solver_d_kwargs': {'max_iter': 300},
    'sort_atoms': True,
    'verbose': 1,
    'random_state': 0,
    'use_batch_cdl': True,
    'n_splits': 10,
    'n_jobs': 5
}
