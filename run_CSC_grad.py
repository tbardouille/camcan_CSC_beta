# Running the CSC analysis from the "mu" example on the alphaCSC website
#       This may not extract beta bursts because the interval is too long
#       but it should find mu bursts

import os
import sys
from pathlib import Path
import numpy as np
import pickle
from scipy.signal import tukey
from scipy.signal.filter_design import maxflat

import mne
from alphacsc import BatchCDL, GreedyCDL
from alphacsc.utils.signal import split_signal


###############################################################################

# if True, use the epoched data file, if False, use ful length data
make_epoch = False

# Important variables
# Define the shape of the dictionary
n_atoms = 25                        # Number of atoms
atomDuration = 0.7  # [seconds]     # Atom duration

# Resample MEG data to this rate for CSC analysis
# sfreq = 300.
sfreq = 150.
sensorType = 'grad'                 # Use this sensor type for CSC analysis
subjectID = 'CC620264'              # Subject to analyse

# compute length of an atom, in timestamps
n_times_atom = int(np.round(atomDuration * sfreq))

# Paths
# homeDir = Path(os.path.expanduser("~"))
homeDir = Path.home()
# inputDir = homeDir / 'data/camcan/proc_data/TaskSensorAnalysis_transdef'
# same path that the one in camcam_process_to_evoked_parallel.py
inputDir = homeDir / 'camcan' / 'proc_data' / 'TaskSensorAnalysis_transdef'
outputDir = homeDir / 'data/CSC'

# Create folders
subjectInputDir = inputDir / subjectID
if not subjectInputDir.exists():
    subjectInputDir.mkdir(parents=True)

subjectOutputDir = outputDir / subjectID
if not subjectOutputDir.exists():
    subjectOutputDir.mkdir(parents=True)


###############################################################################
# Read in an MEG dataset with ~60 trials

print('Reading MEG Data')
fifName = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_'
if make_epoch:
    fifName += 'cleaned-epo.fif'
else:
    fifName += 'cleaned-raw.fif'
megFile = subjectInputDir / fifName

if not megFile.exists():
    sys.exit("Put %s file into %s folder."
             % (fifName, subjectInputDir))

if make_epoch:
    pkl_name = 'CSCepochs_'
else:
    pkl_name = 'CSCraw_'
pkl_name += str(int(atomDuration*1000)) + \
    'ms_' + sensorType + str(n_atoms) + 'atoms.pkl'
outputFile = subjectOutputDir / pkl_name


# Read in the data
if make_epoch:
    epochs = mne.read_epochs(megFile)
    epochs.pick_types(meg=sensorType)

    # Band-pass filter the data to a range of interest
    epochsBandPass = epochs.filter(l_freq=2, h_freq=45)

    # SKIP: notch filter to remove 50 Hz power line noise since
    # LP filter is at 45 Hz
    # epochsNotch = epochsBandPass.filter(l_freq=48, h_freq=52)
    epochsNotch = epochsBandPass

    # Downsample data to 150 Hz to match alphacsc example
    epochsResample = epochsNotch.resample(sfreq=sfreq)

    # Scale data prior to fitting with CSC
    Y = epochsResample.get_data()
    num_trials, num_chans, num_samples = Y.shape
    Y *= tukey(num_samples, alpha=0.1)[None, None, :]
    Y /= np.std(Y)
else:
    raw = mne.io.read_raw_fif(megFile, preload=True)

    # Band-pass filter the data to a range of interest
    raw.filter(l_freq=None, h_freq=125)
    raw.notch_filter([50, 100])

    events = mne.find_events(raw, 'STI101', shortest_event=1)
    raw, events = raw.resample(
        sfreq, npad='auto', verbose=False, events=events)

    X = raw.get_data(picks=['meg'])
    Y = split_signal(X, n_splits=10, apply_window=True)

# epochs.plot()

###############################################################################
# Next, we define the parameters for multivariate CSC

print('Building CSC')

# cdlMEG = BatchCDL(
#     # Shape of the dictionary
#     n_atoms=n_atoms,
#     n_times_atom=n_times_atom,
#     # Request a rank1 dictionary with unit norm temporal and spatial maps
#     rank1=True, uv_constraint='separate',
#     # Initialize the dictionary with random chunk from the data
#     D_init='chunk',
#     # rescale the regularization parameter to be 20% of lambda_max
#     lmbd_max="scaled", reg=.2,
#     # Number of iteration for the alternate minimization and cvg threshold
#     n_iter=100, eps=1e-4,
#     # solver for the z-step
#     solver_z="lgcd", solver_z_kwargs={'tol': 1e-2, 'max_iter': 1000},
#     # solver for the d-step
#     solver_d='alternate_adaptive', solver_d_kwargs={'max_iter': 300},
#     # Technical parameters
#     verbose=1, random_state=0, n_jobs=16)

cdlMEG = GreedyCDL(
    # Shape of the dictionary
    n_atoms=n_atoms,
    n_times_atom=n_times_atom,
    # Request a rank1 dictionary with unit norm temporal and spatial maps
    rank1=True,
    uv_constraint='separate',
    # apply a temporal window reparametrization
    window=True,
    # at the end, refit the activations with fixed support
    # and no reg to unbias
    unbiased_z_hat=True,
    # Initialize the dictionary with random chunk from the data
    D_init='chunk',
    # rescale the regularization parameter to be a percentage of lambda_max
    lmbd_max="scaled",  # original value: "scaled"
    reg=.2,
    # Number of iteration for the alternate minimization and cvg threshold
    n_iter=100,  # original value: 100
    eps=1e-4,  # original value: 1e-4
    # solver for the z-step
    solver_z="lgcd",
    solver_z_kwargs={'tol': 1e-2,  # stopping criteria
                     'max_iter': 1000},
    # solver for the d-step
    solver_d='alternate_adaptive',
    solver_d_kwargs={'max_iter': 300},  # original value: 300
    # sort atoms by explained variances
    sort_atoms=True,
    # Technical parameters
    verbose=1,
    random_state=0,
    n_jobs=5)

###############################################################################
# Fit the model and learn rank1 atoms
print('Running CSC')
cdlMEG.fit(Y)

if make_epoch:
    info = epochsResample.info
else:
    info = X.info

# Save results of CSC with dataset info
pickle.dump([cdlMEG, info], open(outputFile, "wb"))
