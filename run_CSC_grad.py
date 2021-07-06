# Running the CSC analysis from the "mu" example on the alphaCSC website
#       This may not extract beta bursts because the interval is too long
#       but it should find mu bursts

import sys
import numpy as np
import pickle
from scipy.signal import tukey
from joblib import Memory

import mne
from alphacsc import BatchCDL, GreedyCDL
from alphacsc.utils.signal import split_signal

from utils import CACHEDIR, getPaths, getCSCPickleName, getGoodButtonsEvents

memory = Memory(CACHEDIR, verbose=0)


@memory.cache()
def run_csc(subjectID, cdl_on_epoch=True, n_atoms=25, atomDuration=0.7,
            sfreq=150., sensorType='grad',
            use_batch_cdl=False, use_greedy_cdl=True,
            reg=.2, eps=1e-4, tol_z=1e-2):
    """

    Parameters
    ----------
    subjectID : str
        Subject to analyse

    cdl_on_epoch : bool
        if True, use the epoched data file, if False, use ful length data
        default is True

    n_atoms : int
        number of atoms to extract from signals

    atomDuration : float
        atom's duration, in second
        default is 0.7

    sfreq : float
        sampling freuency
        default is 150.

    sensorType : str
        sensor type to use for CSC
        default is 'grad'

    use_batch_cdl, use_greedy_cdl : bool
        decide wether to use alphacsc.BatchCDL or alphacsc.GreedyCDL
        one and only one must be set to True

    reg : float
        regularization parameter for the CSC
        default is 0.2

    eps : float
        CDL  convergence threshold
        default is 1e-4

    tol_z : float
        stopping criteria for Z
        default is 1e-2


    Returns
    -------
    None

    """

    assert (use_batch_cdl + use_greedy_cdl) == 1

    dictPaths = getPaths(subjectID)
    subjectInputDir = dictPaths['procSubjectOutDir']
    subjectOutputDir = dictPaths['cscSubjectOutDir']

    dsPrefix = 'transdef_transrest_mf2pt2_task_raw'

    print('Reading MEG Data')
    if cdl_on_epoch:
        fifName = dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo.fif'
    else:
        fifName = dsPrefix + '_cleaned.fif'
    megFile = subjectInputDir / fifName

    if not megFile.exists():
        sys.exit("Put %s file into %s folder."
                 % (fifName, subjectInputDir))

    # Define CDL results pickle name
    pkl_name = getCSCPickleName(
        use_batch_cdl=use_batch_cdl, use_greedy_cdl=use_greedy_cdl,
        cdl_on_epoch=cdl_on_epoch, atomDuration=atomDuration,
        sensorType=sensorType, n_atoms=n_atoms, reg=reg, eps=eps, tol_z=tol_z)
    outputFile = subjectOutputDir / pkl_name

    # Read and transform the data
    if cdl_on_epoch:
        epochs = mne.read_epochs(megFile)
        epochs.pick_types(meg=sensorType)
        # epochs.plot()

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
        raw.pick_types(meg=sensorType, stim=True)

        # Band-pass filter the data to a range of interest
        raw.filter(l_freq=2, h_freq=45)

        events = mne.find_events(raw, 'STI101', shortest_event=1)
        raw, events = raw.resample(
            sfreq, npad='auto', verbose=False, events=events)

        _, goodButtonEvents = getGoodButtonsEvents(
            raw, stim_channel='STI101', subtract_first_samp=True)
        mne.write_events(subjectOutputDir /
                         (dsPrefix + '_Under2SecResponseOnly-eve.fif'),
                         goodButtonEvents)

        X = raw.get_data(picks=['meg'])
        Y = split_signal(X, n_splits=10, apply_window=True)

    ###########################################################################
    # Next, we define the parameters for multivariate CSC

    print('Building CSC')

    # compute length of an atom, in timestamps
    n_times_atom = int(np.round(atomDuration * sfreq))

    cdlParams = {
        # Shape of the dictionary
        'n_atoms': n_atoms,
        'n_times_atom': n_times_atom,
        # Request a rank1 dictionary with unit norm temporal and spatial maps
        'rank1': True, 'uv_constraint': 'separate',
        # Apply a temporal window reparametrization
        'window': True,  # in Tim's: False
        # At the end, refit the activations with fixed support
        # and no reg to unbias
        'unbiased_z_hat': True,  # in Tim's: False
        # Initialize the dictionary with random chunk from the data
        'D_init': 'chunk',
        # Rescale the regularization parameter to be 20% of lambda_max
        'lmbd_max': 'scaled', 'reg': reg,
        # Number of iteration for the alternate minimization and cvg threshold
        'n_iter': 100, 'eps': eps,
        # Solver for the z-step
        'solver_z': 'lgcd',
        'solver_z_kwargs': {'tol': tol_z,
                            'max_iter': 1000},
        # Solver for the d-step
        'solver_d': 'alternate_adaptive',
        'solver_d_kwargs': {'max_iter': 300},
        # Sort atoms by explained variances
        'sort_atoms': True,
        # Technical parameters
        'verbose': 1, 'random_state': 0, 'n_jobs': 5}

    if use_batch_cdl:
        cdlMEG = BatchCDL(**cdlParams)
    elif use_greedy_cdl:
        cdlMEG = GreedyCDL(**cdlParams)

    ###########################################################################
    # Fit the model and learn rank1 atoms
    print('Running CSC')
    cdlMEG.fit(Y)

    if cdl_on_epoch:
        info = epochsResample.info
        z_hat_ = cdlMEG.z_hat_
    else:
        print("Compute atoms' activation on full data")
        z_hat_ = cdlMEG.transform(X[None, :])
        info = raw.copy().pick_types(meg=True).info

    # Save results of CSC with dataset info
    res = [cdlMEG, info, z_hat_]
    pickle.dump(res, open(outputFile, "wb"))

    return res


if __name__ == '__main__':
    run_csc(subjectID='CC620264', cdl_on_epoch=False, n_atoms=25,
            atomDuration=0.7, sfreq=150.,
            use_batch_cdl=True, use_greedy_cdl=False,
            reg=.2, eps=1e-4, tol_z=1e-2)
