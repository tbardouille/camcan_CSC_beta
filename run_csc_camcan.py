# %%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Memory, hash
import json

import mne
from mne_bids import BIDSPath, read_raw_bids
from alphacsc.viz.epoch import make_epochs

from utils_csc import run_csc
from utils_plot import plot_csc
from utils_dripp import get_dripp_results

DATA_DIR = Path("/storage/store/data/")
BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"  # Root path to raw BIDS files
SSS_CAL_FILE = DATA_DIR / "camcan-mne/Cam-CAN_sss_cal.dat"
CT_SPARSE_FILE = DATA_DIR / "camcan-mne/Cam-CAN_ct_sparse.fif"
PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"
HOME_DIR = Path('.')

mem = Memory('.')

participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t', header=0)

subject_id = "CC620264"

if len(sys.argv) > 1:  # get subject_id from command line
    try:
        subject_id_idx = int(sys.argv[1])
        subject_id = participants.iloc[subject_id_idx]['participant_id']
        subject_id = subject_id.split('-')[1]
    except ValueError:
        pass

age, sex = participants[participants['participant_id']
                        == 'sub-' + str(subject_id)][['age', 'sex']].iloc[0]

print(f'Running CSC pipeline on: {subject_id}, {str(age)} year old {sex}')
fig_title = f'Subject {subject_id}, {str(age)} year old {sex}'

# %% Parameters
ch_type = "grad"  # run CSC
sfreq = 150.

# Epoching parameters
tmin = -1.7
tmax = 1.7
baseline = (-1.25, -1.0)

activation_tstart = -tmin
shift_acti = False  # put activation to the peak amplitude time in the atom

exp_params = {
    "subject_id": subject_id,
    "sfreq": sfreq,  # in Tim's: 300
    "atom_duration": 0.7,  # in Tim's: 0.5,
    "n_atoms": 30,  # in Tim's: 25,
    "reg": 0.2,  # in Tim's: 0.2,
    "eps": 1e-5,  # in Tim's: 1e-4,
    "tol_z": 1e-3,  # in Tim's: 1e-2
}

cdl_params = {
    'n_atoms': exp_params['n_atoms'],
    'n_times_atom': int(np.round(exp_params["atom_duration"] * sfreq)),
    'rank1': True, 'uv_constraint': 'separate',
    'window': True,  # in Tim's: False
    'unbiased_z_hat': True,  # in Tim's: False
    'D_init': 'chunk',
    'lmbd_max': 'scaled', 'reg': exp_params['reg'],
    'n_iter': 100, 'eps': exp_params['eps'],
    'solver_z': 'lgcd',
    'solver_z_kwargs': {'tol': exp_params['tol_z'], 'max_iter': 1000},
    'solver_d': 'alternate_adaptive',
    'solver_d_kwargs': {'max_iter': 300},
    'sort_atoms': True,
    'verbose': 1,
    'random_state': 0,
    'use_batch_cdl': True,
    'n_splits': 10,
    'n_jobs': 5
}

dripp_params = {
    'threshold': 0.5e-10,
    'lower': 0, 'upper': 500e-3,
    'sfreq': sfreq,
    'initializer': 'smart_start',
    'alpha_pos': True,
    'n_iter': 80,
    'verbose': False,
    'disable_tqdm': False
}

all_params = [exp_params, cdl_params, dripp_params]

# Create folder to save results for the considered subject
subject_output_dir = HOME_DIR / "results" / subject_id
subject_output_dir.mkdir(parents=True, exist_ok=True)
# Create folder to save final figures for a particular set of parameters
exp_output_dir = subject_output_dir / hash(all_params)
exp_output_dir.mkdir(parents=True, exist_ok=True)
# Save experiment parameters
with open(exp_output_dir / 'exp_params', 'w') as fp:
    json.dump(all_params, fp, sort_keys=True, indent=4)

# %% Read raw data from BIDS file
bp = BIDSPath(
    root=BIDS_ROOT,
    subject=subject_id,
    task="smt",
    datatype="meg",
    extension=".fif",
    session="smt",
)
raw = read_raw_bids(bp)

# %% Preproc data

raw.load_data()
raw.filter(l_freq=None, h_freq=125)
raw.notch_filter([50, 100])
raw = mne.preprocessing.maxwell_filter(raw, calibration=SSS_CAL_FILE,
                                       cross_talk=CT_SPARSE_FILE,
                                       st_duration=10.0)

# %% Now deal with Epochs

all_events, all_event_id = mne.events_from_annotations(raw)
# all_event_id = {'audiovis/1200Hz': 1,
#                 'audiovis/300Hz': 2,
#                 'audiovis/600Hz': 3,
#                 'button': 4,
#                 'catch/0': 5,
#                 'catch/1': 6}

# for every button event,
metadata_tmin, metadata_tmax = -3., 0
row_events = ['button']
keep_last = ['audiovis']

metadata, events, event_id = mne.epochs.make_metadata(
    events=all_events, event_id=all_event_id,
    tmin=metadata_tmin, tmax=metadata_tmax, sfreq=raw.info['sfreq'],
    row_events=row_events, keep_last=keep_last)

epochs = mne.Epochs(
    raw, events, event_id, metadata=metadata,
    tmin=tmin, tmax=tmax,
    baseline=baseline,
    preload=True, verbose=False
)

# "good" button events in Tim's:
# button event is at most one sec. after an audiovis event,
# and with at least 3 sec. between 2 button events.
epochs = epochs["event_name == 'button' and audiovis > -1. and button == 0."]
# epochs = epochs["event_name == 'button' and audiovis > -3."]

# XXX
if True:
    evokeds = epochs.average()
    fig = evokeds.plot_joint(picks="grad")
    fig_name = "evokeds.pdf"
    fig.savefig(subject_output_dir / fig_name, dpi=300)

# %% Setup CSC on Raw

raw_csc = raw.copy()  # make a copy to run CSC on ch_type
raw_csc.pick([ch_type, 'stim'])

# Band-pass filter the data to a range of interest
raw_csc.filter(l_freq=2, h_freq=45)
raw_csc, events = raw_csc.resample(
    sfreq, npad='auto', verbose=False, events=epochs.events)

X = raw_csc.get_data(picks=['meg'])

# %% Run multivariate CSC

cdl_model, z_hat_ = mem.cache(run_csc)(X, **cdl_params)

# %% Get and plot CSC results

print("Get CSC results")

# events here are only "good" button events
events_no_first_samp = events.copy()
events_no_first_samp[:, 0] -= raw_csc.first_samp
info = raw_csc.info.copy()
info["events"] = events_no_first_samp
info["event_id"] = None
# atom_duration = exp_params.get("atom_duration", 0.7)
allZ = make_epochs(
    z_hat_,
    info,
    t_lim=(-activation_tstart, activation_tstart),
    # n_times_atom=int(np.round(atom_duration * sfreq)),
    n_times_atom=cdl_params['n_times_atom'])

# %%
df_dripp = get_dripp_results(cdl_model,
                             z_hat_,
                             sfreq,
                             events=events_no_first_samp,
                             event_id=all_event_id['button'],
                             dripp_params=dripp_params,
                             save_dir=exp_output_dir)

# %%

plot_csc(cdl_model=cdl_model,
         raw_csc=raw_csc,
         allZ=allZ,
         plot_acti_histo=True,
         activation_tstart=activation_tstart,
         df_dripp=df_dripp,
         save_dir=exp_output_dir,
         title=fig_title)


# %%
