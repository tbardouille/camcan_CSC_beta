import numpy as np
from pathlib import Path


TEAM = 'parietal'  # 'dal' | 'parietal' | 'cedric'

if TEAM == 'parietal':
    # path to CSC results
    RESULTS_DIR = Path('./results_csc')
    # Paths for Cam-CAN dataset
    DATA_DIR = Path("/storage/store/data/")
    BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"  # root path to raw BIDS files
    SSS_CAL_FILE = DATA_DIR / "camcan-mne/Cam-CAN_sss_cal.dat"
    CT_SPARSE_FILE = DATA_DIR / "camcan-mne/Cam-CAN_ct_sparse.fif"
    PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"
    HOME_DIR = Path('.')
    N_JOBS = 20

elif TEAM == 'dal':
    # path to CSC results
    RESULTS_DIR = Path('/media/NAS/lpower/CSC/results/')
    # Paths for Cam-CAN dataset
    DATA_DIR = Path(
        "/media/WDEasyStore/timb/camcan/release05/cc700/meg/pipeline/release005/")
    BIDS_ROOT = DATA_DIR / "BIDSsep/smt/"  # root path to raw BIDS files
    SSS_CAL_FILE = Path(
        "/home/timb/camcan/camcanMEGcalibrationFiles/sss_cal.dat")
    CT_SPARSE_FILE = Path(
        "/home/timb/camcan/camcanMEGcalibrationFiles/ct_sparse.fif")
    PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"
    HOME_DIR = Path("/media/NAS/lpower/CSC/")
    N_JOBS = 6

elif TEAM == 'cedric':
    # path to CSC results
    RESULTS_DIR = Path('./results_csc')
    PARTICIPANTS_FILE = Path("./participants.tsv")
    N_JOBS = 6

try:
    # list of all Can-CAN subject ids
    SUBJECT_IDS = [f.name.split('-')[1] for f in BIDS_ROOT.iterdir() if
                   (not f.is_file()) and (f.name[:6] == 'sub-CC')]
except (FileNotFoundError):
    print(f'No such file or directory: {BIDS_ROOT}')
    SUBJECT_IDS = []


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
        BEM_DIR = DATA_DIR / "camcan-mne/freesurfer"
        TRANS_DIR = DATA_DIR / "camcan-mne/trans"

        # XXX find the good paths for Parital drago server
        bemFif = BEM_DIR / subject_id / \
            'bem' / (subject_id + '-meg-bem.fif')
        transFif = TRANS_DIR / ('sub-' + subject_id + '-trans.fif')
        if not transFif.exists():
            transFif = Path(str(transFif).replace('trans/', 'trans-halifax/'))
        # XXX
        EPOCH_DIR = ''
        fifFile = ''
        epochFif = EPOCH_DIR + subject_id + fifFile

    return epochFif, transFif, bemFif


# hare all the global variables
EXP_PARAMS = {
    "sfreq": 150.,             # 300
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
    'window': True,
    'unbiased_z_hat': True,
    'D_init': 'chunk',
    'lmbd_max': 'scaled',
    'reg': EXP_PARAMS['reg'],
    'n_iter': 100,
    'eps': EXP_PARAMS['eps'],
    'solver_z': 'lgcd',
    'solver_z_kwargs': {'tol': EXP_PARAMS['tol_z'],
                        'max_iter': 1000},
    'solver_d': 'alternate_adaptive',
    'solver_d_kwargs': {'max_iter': 300},
    'sort_atoms': True,
    'verbose': 1,
    'random_state': 0,
    'use_batch_cdl': True,
    'n_splits': 10,
    'n_jobs': 5
}


def get_cdl_pickle_name():
    return 'CSCraw_' + str(EXP_PARAMS['atom_duration']) + 's_' + \
        str(CDL_PARAMS['n_atoms']) + 'atoms.pkl'
