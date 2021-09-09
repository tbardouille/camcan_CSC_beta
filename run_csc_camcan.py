# %%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Memory
import mne
from mne_bids import BIDSPath, read_raw_bids
# from alphacsc import BatchCDL, GreedyCDL
# from alphacsc.utils.signal import split_signal
from alphacsc.viz.epoch import make_epochs

from utils_csc import run_csc
from utils_plot import plot_csc

DATA_DIR = Path("/storage/store/data/")
BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"  # Root path to raw BIDS files
SSS_CAL_FILE = DATA_DIR / "camcan-mne/Cam-CAN_sss_cal.dat"
CT_SPARSE_FILE = DATA_DIR / "camcan-mne/Cam-CAN_ct_sparse.fif"
PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"

participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t', header=0)

subject_id = "CC620264"

if len(sys.argv) > 1:  # get subject_id from command line
    subject_id_idx = int(sys.argv[1])
    subject_id = participants.iloc[subject_id_idx]['participant_id']
    subject_id = subject_id.split('-')[1]

# mem = Memory('.')

print(f'Running CSC pipeline on: {subject_id}')

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

HOME_DIR = Path('.')

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
    'use_batch_cdl': False,
    'n_splits': 10,
    'n_jobs': 5
}

# Create folder to save results
subject_output_dir = HOME_DIR / "results" / subject_id
subject_output_dir.mkdir(parents=True, exist_ok=True)

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
# print('Computing CSC')


# @mem.cache
# def run_csc(X, **cdl_params):
#     cdl_params = dict(cdl_params)
#     n_splits = cdl_params.pop('n_splits', 10)
#     use_batch_cdl = cdl_params.pop('use_batch_cdl', False)
#     if use_batch_cdl:
#         cdl_model = BatchCDL(**cdl_params)
#     else:
#         cdl_model = GreedyCDL(**cdl_params)

#     X_splits = split_signal(X, n_splits=n_splits, apply_window=True)

#     # Fit the model and learn rank1 atoms
#     print('Running CSC')
#     cdl_model.fit(X_splits)

#     z_hat_ = cdl_model.transform(X[None, :])
#     return cdl_model, z_hat_


cdl_model, z_hat_ = run_csc(X, **cdl_params)

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

plot_csc(n_atoms_est=z_hat_.shape[1],
         atom_duration=exp_params["atom_duration"],
         cdl_model=cdl_model,
         info=info,
         sfreq=sfreq,
         plot_acti_histo=True,
         allZ=allZ,
         activation_tstart=activation_tstart,
         save_dir=subject_output_dir)


# fontsize = 12
# n_atoms_est = z_hat_.shape[1]
# n_atoms_per_fig = 5
# figsize = (15, 7)

# atoms_in_figs = np.arange(0, n_atoms_est + 1, n_atoms_per_fig)
# atoms_in_figs = list(zip(atoms_in_figs[:-1], atoms_in_figs[1:]))

# for fig_idx, (atoms_start, atoms_stop) in enumerate(atoms_in_figs, start=1):
#     fig, axes = plt.subplots(4, n_atoms_per_fig, figsize=figsize)

#     for i_atom, kk in enumerate(range(atoms_start, atoms_stop)):
#         ax = axes[0, i_atom]
#         ax.set_title("Atom #" + str(kk), fontsize=fontsize)

#         # Spatial pattern
#         u_hat = cdl_model.u_hat_[kk]
#         mne.viz.plot_topomap(u_hat, info, axes=ax, show=False)
#         if i_atom == 0:
#             ax.set_ylabel("Spatial", labelpad=86, fontsize=fontsize)

#         # Temporal pattern
#         ax = axes[1, i_atom]
#         v_hat = cdl_model.v_hat_[kk]
#         t = np.arange(v_hat.size) / sfreq
#         ax.plot(t, v_hat)
#         ax.grid(True)
#         ax.set_xlim(0, atom_duration)  # crop x axis
#         if i_atom == 0:
#             ax.set_ylabel("Temporal", labelpad=14, fontsize=fontsize)

#         # Power Spectral Density (PSD)
#         ax = axes[2, i_atom]
#         psd = np.abs(np.fft.rfft(v_hat, n=256)) ** 2
#         frequencies = np.linspace(0, sfreq / 2.0, len(psd))
#         ax.semilogy(frequencies, psd, label="PSD", color="k")
#         ax.set_xlim(0, 40)  # crop x axis
#         ax.set_xlabel("Frequencies (Hz)", fontsize=fontsize)
#         ax.grid(True)
#         if i_atom == 0:
#             ax.set_ylabel("Power Spectral Density", labelpad=13,
#                           fontsize=fontsize)

#         # Atom's activations
#         ax = axes[3, i_atom]
#         z_hat = allZ[:, i_atom, :]
#         if shift_acti:
#             # roll to put activation to the peak amplitude time in the atom
#             shift = np.argmax(np.abs(cdl_model.v_hat_[i_atom]))
#             z_hat = np.roll(z_hat, shift, axis=1)
#             z_hat[:, :shift] = 0  # pad with 0
#         # t1 = np.arange(cdl.z_hat_.shape[2]) / sfreq - 1.7
#         t1 = np.arange(allZ.shape[2]) / sfreq - activation_tstart
#         ax.plot(t1, z_hat.T)
#         ax.set_xlabel("Time (s)", fontsize=fontsize)

#         if i_atom == 0:
#             ax.set_ylabel("Atom's activations", labelpad=7, fontsize=fontsize)

#         fig.tight_layout()

#         fig_name = f"atoms_part_{fig_idx}.pdf"
#         fig.savefig(subject_output_dir / fig_name, dpi=300)
#         fig.savefig(subject_output_dir / (fig_name.replace(".pdf", ".png")),
#                     dpi=300)
#         # fig.close()

# plt.show()
