"""
Script for all utils functions, from data-preprocessing, to CSC model and
figure plot.
"""

# === camcan event description ===

# Label:          Auditory 300Hz
# Event Type:     STI101_up
# Event code:     6
# Delay:          13

# Label:          Auditory 600Hz
# Event Type:     STI101_up
# Event code:     7
# Delay:          13

# Label:          Auditory 1200Hz
# Event Type:     STI101_up
# Event code:     8
# Delay:          13

# Label:          Visual Checkerboard
# Event Type:     STI101_up
# Event code:     9
# Delay:          34

###############################################################################
# PACKAGES
###############################################################################

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from joblib import Memory
from scipy.signal import tukey

import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs

from alphacsc import BatchCDL, GreedyCDL
from alphacsc.utils.signal import split_signal

###############################################################################
# GLOBAL VARIABLES
###############################################################################

DATA_DIR = Path("/storage/store/data/")
BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"  # Root path to raw BIDS files

SSS_CAL_FILE = DATA_DIR / "camcan-mne/Cam-CAN_sss_cal.dat"
CT_SPARSE_FILE = DATA_DIR / "camcan-mne/Cam-CAN_ct_sparse.fif"
PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"


# Define the cache directory for joblib.Memory
CACHEDIR = Path('./__cache__')
memory = Memory(CACHEDIR, verbose=0)

###############################################################################
# PRE-PROCESSING
###############################################################################

# Script to take raw MEG files from CamCAN and generate evoked response, with
# some other files along the way


def get_good_buttons_events(raw, stim_channel='STI101',
                            subtract_first_samp=True):
    """ Find button events from raw data that satisfy some conditions.
    Buttons events are the events with id > 10.
    A "good" button event is a button event occuring less than one second
    after a "regular" event and more than 3 seconds after the previous button
    event.

    Parameters
    ----------
    raw : instance of mne.Raw
        raw data

    stim_channel : string
        name of the STIM channel
        default is 'STI101'

    subtract_first_samp : bool
        if True, substract raw.first_samp to events onsets

    Returns
    -------
    events : array-like of shape (n_events, 3)
        all events

    good_buttons_events : array-like of shape (n_button_events, 3)
        only "good" button events
        the ID (third column) is set to 128 for all events

    """
    # Find button presses to stimuli
    events = mne.find_events(raw, stim_channel, shortest_event=1)
    if subtract_first_samp:
        events[:, 0] -= raw.first_samp
    # Get stimuli and response latencies
    # Pull event IDs
    event_id = events[:, 2]
    # Get all events with ID < 10 (cues)
    stim_events = events[np.where(event_id < 10)[0], :]
    stim_onsets = stim_events[:, 0]
    # Get all events with ID > 10 (button press) - not always the same number
    button_events = events[np.where(event_id > 10)[0], :]
    # Make the button press event always have ID = 128
    button_onsets = button_events[:, 0]
    button_events[:, 2] = 128
    # Stimulus loop to find the next button press under 2 seconds
    good_button_events = []
    # Loop per cue
    for this_stim_sample in stim_onsets:
        # Find timing of responses wrt stimulus
        all_rt = button_onsets - this_stim_sample
        # Find where this timing is positive
        positive_rt_index = np.where(all_rt > 0)[0]
        # If there is a positive response timing ...
        if len(positive_rt_index) > 0:
            # And if that positive timing is less than 1 second ...
            this_rt = all_rt[positive_rt_index][0] / raw.info['sfreq']
            if this_rt < 1:
                # Then also check that this is the first button press, or the
                # previous response was more than 3 seconds ago
                this_button_press_event = button_events[positive_rt_index[0], :]
                this_onset = this_button_press_event[0]
                relative_button_samples = button_onsets - this_onset
                prior_bp_index = np.where(relative_button_samples < 0)[0]
                # If this is the first button press
                if len(prior_bp_index) == 0:
                    if len(good_button_events) == 0:
                        good_button_events = this_button_press_event
                    else:
                        good_button_events = np.vstack(
                            (good_button_events, this_button_press_event))
                else:
                    # If not, check the time from previous response
                    samples_to_prior_response = \
                        relative_button_samples[prior_bp_index[-1]]
                    time_to_prior_response = -1 * \
                        samples_to_prior_response / raw.info['sfreq']
                    if time_to_prior_response > 3:
                        # Then either start a matrix with good button press
                        # events, or add to it
                        if len(good_button_events) == 0:
                            good_button_events = this_button_press_event
                        else:
                            good_button_events = np.vstack(
                                (good_button_events, this_button_press_event))

    if len(good_button_events) == 0:
        print("No 'good' button event found.")
    else:
        # Drop duplicate events in the button press list
        events_df = pd.DataFrame(good_button_events)
        good_button_events = events_df.drop_duplicates().values

    return events, good_button_events


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())


@memory.cache()
def meg_preproc(subject_id, apply_maxwell_filter=True,
                apply_ica_cleaning=False):
    """

    Parameters
    ----------
    subject_id : str
        the subject ID we are interested in

    apply_maxwell_filter : bool
        if True, apply a Maxwell filter
        default is False (as Tim's file is pre-filtered)

    apply_ica_cleaning : bool
        XXX

    Returns
    -------
    epochs, evoked, raw

    """

    # Read raw BIDS file associated to desired subject
    bp = BIDSPath(root=BIDS_ROOT, subject=subject_id,
                  task="smt", datatype="meg", extension=".fif", session="smt")
    raw = read_raw_bids(bp)

    # Filter data
    raw.filter(l_freq=None, h_freq=125)
    raw.notch_filter([50, 100])
    if apply_maxwell_filter:
        raw = mne.preprocessing.maxwell_filter(raw, calibration=SSS_CAL_FILE,
                                               cross_talk=CT_SPARSE_FILE,
                                               st_duration=10.0)

    # Get all events and "good" button events
    events, good_button_events = get_good_buttons_events(
        raw, stim_channel='STI101', subtract_first_samp=True)

    # Epoching parameters
    prestim = -1.7
    poststim = 1.7
    base_start = -1.25
    base_end = -1.0

    epochs = mne.Epochs(raw, events=np.array(good_button_events),
                        event_id=None,
                        tmin=prestim, tmax=poststim,
                        baseline=(base_start, base_end),
                        verbose=False, preload=True)

    if apply_ica_cleaning:
        print('Running ICA')
        reject = dict(grad=4000e-13, mag=5e-12)
        picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True,
                               stim=False, exclude='bads')
        ica = ICA(n_components=0.99, method='fastica')
        ica.fit(raw, picks=picks, reject=reject)

        n_max_ecg, n_max_eog = 3, 3

        # Reject bad EOG components following mne procedure
        try:
            eog_epochs = create_eog_epochs(
                raw, tmin=-0.5, tmax=0.5, reject=reject)
            eog_inds, scores = ica.find_bads_eog(eog_epochs)
            eog_inds = eog_inds[:n_max_eog]
            ica.exclude.extend(eog_inds)
        except:
            print(
                """Subject {0} had no eog/eeg channels""".format(
                    str(subject_id)))

        # Reject bad ECG compoments following mne procedure
        ecg_epochs = create_ecg_epochs(raw, tmin=-0.5, tmax=0.5)
        ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
        ecg_inds = ecg_inds[:n_max_ecg]
        ica.exclude.extend(ecg_inds)
        # save ICA file
        # ica.save(icaFif)
        # apply ICA to epoched and raw data
        ica.apply(epochs, exclude=ica.exclude)
        ica.apply(raw, exclude=ica.exclude)
        # Print some info
        print("Subject ID: %s\nEpochs length: %i\nICA exclude: %s" %
              (str(subject_id), len(epochs), str(len(ica.exclude))))

    # Make evoked
    evoked = epochs.average()

    return epochs, evoked, raw


###############################################################################
# RUN CDL MODEL
###############################################################################

@memory.cache()
def run_csc(subject_id, cdl_on_epoch=True,
            n_atoms=25, atomDuration=0.7,
            sfreq=150., sensorType='grad',
            use_batch_cdl=False, use_greedy_cdl=True,
            reg=.2, eps=1e-4, tol_z=1e-2):
    """

    Parameters
    ----------
    subject_id : str
        the subject ID we are interested in

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
    res = [cdl_meg, info, z_hat_]

    """

    assert (use_batch_cdl + use_greedy_cdl) == 1

    epochs, evoked, raw = meg_preproc(subject_id=subject_id,
                                      apply_maxwell_filter=True,
                                      apply_ica_cleaning=False)

    # Read and transform the data
    if cdl_on_epoch:
        epochs.pick_types(meg=sensorType)
        # epochs.plot()
        # Band-pass filter the data to a range of interest
        epochs_band_pass = epochs.filter(l_freq=2, h_freq=45)
        # SKIP: notch filter to remove 50 Hz power line noise since
        # LP filter is at 45 Hz
        # epochs_notch = epochs_band_pass.filter(l_freq=48, h_freq=52)
        epochs_notch = epochs_band_pass
        # Downsample data to 150 Hz to match alphacsc example
        epochs_resample = epochs_notch.resample(sfreq=sfreq)
        # Scale data prior to fitting with CSC
        Y = epochs_resample.get_data()
        num_trials, num_chans, num_samples = Y.shape
        Y *= tukey(num_samples, alpha=0.1)[None, None, :]
        Y /= np.std(Y)
    else:
        raw.pick_types(meg=sensorType, stim=True)
        # Band-pass filter the data to a range of interest
        raw.filter(l_freq=2, h_freq=45)
        # Resample the data
        events = mne.find_events(raw, 'STI101', shortest_event=1)
        raw, events = raw.resample(
            sfreq, npad='auto', verbose=False, events=events)

        X = raw.get_data(picks=['meg'])
        Y = split_signal(X, n_splits=10, apply_window=True)

    # Define the parameters for multivariate CSC
    print('Building CSC')
    cdlParams = {
        # Shape of the dictionary
        'n_atoms': n_atoms,
        # compute length of an atom, in timestamps
        'n_times_atom': int(np.round(atomDuration * sfreq)),
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
        cdl_meg = BatchCDL(**cdlParams)
    elif use_greedy_cdl:
        cdl_meg = GreedyCDL(**cdlParams)

    # Fit the model and learn rank1 atoms
    print('Running CSC')
    cdl_meg.fit(Y)

    if cdl_on_epoch:
        info = epochs_resample.info
        z_hat_ = cdl_meg.z_hat_
    else:
        print("Compute atoms' activation on full data")
        z_hat_ = cdl_meg.transform(X[None, :])
        info = raw.copy().pick_types(meg=True).info

    # Save results of CSC with dataset info
    res = [cdl_meg, info, z_hat_]

    return res


###############################################################################
# RUN DRIPP MODEL ON EXTRACTED ATOMS
###############################################################################

# XXX

###############################################################################
# PLOT FINAL FIGURE
###############################################################################


def get_subject_age(subject_id):
    """
    XXX

    Parameters
    ----------
    subject_id : str
        the subject ID we are interested in

    Returns
    -------
    the age of the considered subject

    """
    participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t', header=0)
    age = participants[participants['participant_id']
                       == 'sub-' + str(subject_id)]['age'].iloc[0]

    return age


def get_exp_name(use_batch_cdl=True, use_greedy_cdl=False,
                 cdl_on_epoch=False, atomDuration=0.7, sensorType='grad',
                 n_atoms=25, reg=.2, eps=1e-4, tol_z=1e-2):
    """

    """
    assert (use_batch_cdl + use_greedy_cdl) == 1

    if use_batch_cdl:
        exp_name = 'Batch'
    elif use_greedy_cdl:
        exp_name = 'Greedy'

    if cdl_on_epoch:
        exp_name += 'CSCepochs_'
    else:
        exp_name += 'CSCraw_'

    exp_name += str(int(atomDuration * 1000)) + \
        'ms_' + sensorType + str(n_atoms) + 'atoms_' + \
        str(reg) + 'reg' + str(eps) + 'eps' + str(tol_z) + 'tol_z'

    return exp_name


def plot_csc(subject_id, cdl_on_epoch=True, n_atoms=25, atomDuration=0.7,
             sfreq=150., sensorType='grad',
             use_batch_cdl=False, use_greedy_cdl=True,
             reg=.2, eps=1e-4, tol_z=1e-2,
             activeStartTime=1.7, shift_acti=False, use_drago=False):
    """

    Parameters
    ----------
    subject_id : str
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

    activeStartTime : float
        Epoching parameters when cdl_on_epoch is False,
        seconds from start of trial

    shift_acti : bool
        if True, shift the atom's activation to put activation to the peak
        amplitude time in the atom
        default is False

    Returns
    -------

    """

    assert (use_batch_cdl + use_greedy_cdl) == 1

    cdl, info, z_hat_ = run_csc(
        subject_id=subject_id, cdl_on_epoch=cdl_on_epoch, n_atoms=n_atoms,
        atomDuration=atomDuration, sfreq=sfreq, sensorType=sensorType,
        use_batch_cdl=use_batch_cdl, use_greedy_cdl=use_greedy_cdl,
        reg=reg, eps=eps, tol_z=tol_z)

    # dsPrefix = 'transdef_transrest_mf2pt2_task_raw'

    # dictPaths = getPaths(subject_id, use_drago=use_drago)
    # subjectInputDir = dictPaths['procSubjectOutDir']
    # subjectOutputDir = dictPaths['cscSubjectOutDir']

    # # Load in the CSC results
    # pkl_name = getCSCPickleName(
    #     use_batch_cdl=use_batch_cdl, use_greedy_cdl=use_greedy_cdl,
    #     cdl_on_epoch=cdl_on_epoch, atomDuration=atomDuration,
    #     sensorType=sensorType, n_atoms=n_atoms, reg=reg, eps=eps, tol_z=tol_z)
    # outputFile = subjectOutputDir / pkl_name

    # res = pickle.load(open(outputFile, "rb"))
    # if len(res) == 2:
    #     cdl, info = res
    #     z_hat_ = cdl.z_hat_
    # elif len(res) == 3:
    #     cdl, info, z_hat_ = res

    # sfreq = info['sfreq']

    # [seconds from start of trial, i.e., -1.7 seconds wrt cue]
    preStimStartTime = 0
    zWindowDurationTime = 0.5   # in seconds
    preStimStart = int(np.round(preStimStartTime * sfreq))
    activeStart = int(np.round(activeStartTime * sfreq))
    zWindowDuration = int(np.round(zWindowDurationTime * sfreq))

    # Define figures names suffixes
    subjectAge = get_subject_age(subject_id=subject_id, use_drago=use_drago)
    figNameSuffix = '_' + pkl_name.replace('.pkl',
                                           '_age' + str(subjectAge) + '.pdf')

    # get effective n_atoms
    n_atoms = z_hat_.shape[1]
    if cdl_on_epoch:
        allZ = z_hat_
    else:
        # transform z_hat_ into epoched, with shape (n_events, n_atoms, n_tt)
        # file with good button events
        eveFif_button = subjectOutputDir / \
            (dsPrefix + '_Under2SecResponseOnly-eve.fif')
        goodButtonEvents = mne.read_events(eveFif_button)
        print("Number of events: %i" % len(goodButtonEvents))

        info['events'] = np.array(goodButtonEvents)
        info['event_id'] = None
        allZ = make_epochs(z_hat_, info,
                           t_lim=(-activeStartTime, activeStartTime),
                           n_times_atom=int(np.round(atomDuration * sfreq)))

        print("allZ shape:", allZ.shape)

    # Calculate the sum of all Z values for prestim and active intervals
    activezHat = allZ[:, :, activeStart: activeStart + zWindowDuration]
    preStimzHat = allZ[:, :, preStimStart: preStimStart + zWindowDuration]
    activeSum = np.sum(np.sum(activezHat, axis=2), 0)
    preStimSum = np.sum(np.sum(preStimzHat, axis=2), 0)
    diffZ = activeSum - preStimSum

    # Plot the change in activation
    plt.figure()
    plt.bar(x=range(n_atoms), height=diffZ)
    plt.ylabel('Aggregate Activation Change')
    plt.xlabel('Atom #')
    plt.savefig(subjectOutputDir / ('change_in_activation' + figNameSuffix),
                dpi=300)
    plt.close()

    # T-test of z-value sums across trials between intervals
    activeSumPerTrial = np.sum(activezHat, axis=2)
    preStimSumPerTrial = np.sum(preStimzHat, axis=2)
    diffZperTrial = activeSumPerTrial-preStimSumPerTrial
    tstat, pvalue = ss.ttest_1samp(diffZperTrial, popmean=0, axis=0)

    # Plot result of the t-test
    plt.figure()
    plt.bar(x=range(n_atoms), height=tstat)
    plt.grid(True)
    plt.ylabel('t-stat')
    plt.xlabel('Atom #')
    plt.savefig(subjectOutputDir / ('t_test' + figNameSuffix), dpi=300)
    plt.close()

    # Plot general figure: spatial & temporal pattern, power spectral density
    # (PSD) and activations

    fontsize = 16

    n_plots = 4
    n_columns = min(5, n_atoms)
    split = int(np.ceil(n_atoms / n_columns))
    figsize = (4 * n_columns, 3 * n_plots * split)
    fig, axes = plt.subplots(n_plots * split, n_columns, figsize=figsize)

    for i_atom, kk in enumerate(range(n_atoms)):
        i_row, i_col = i_atom // n_columns, i_atom % n_columns
        it_axes = iter(axes[i_row * n_plots:(i_row + 1) * n_plots, i_col])

        ax = next(it_axes)
        ax.set_title('Atom #' + str(i_atom), fontsize=fontsize)

        # Spatial pattern
        u_hat = cdl.u_hat_[i_atom]
        mne.viz.plot_topomap(u_hat, info, axes=ax, show=False)
        if i_col == 0:
            ax.set_ylabel('Spatial', labelpad=86, fontsize=fontsize)

        # Temporal pattern
        ax = next(it_axes)
        v_hat = cdl.v_hat_[i_atom]
        t = np.arange(v_hat.size) / sfreq
        ax.plot(t, v_hat)
        ax.grid(True)
        ax.set_xlim(0, atomDuration)  # crop x axis
        if i_col == 0:
            ax.set_ylabel('Temporal', labelpad=14, fontsize=fontsize)

        # Power Spectral Density (PSD)
        ax = next(it_axes)
        psd = np.abs(np.fft.rfft(v_hat, n=256)) ** 2
        frequencies = np.linspace(0, sfreq / 2.0, len(psd))
        ax.semilogy(frequencies, psd, label='PSD', color='k')
        ax.set_xlim(0, 40)  # crop x axis
        ax.set_xlabel('Frequencies (Hz)', fontsize=fontsize)
        ax.grid(True)
        if i_col == 0:
            ax.set_ylabel('Power Spectral Density',
                          labelpad=13, fontsize=fontsize)

        # Atom's activations
        ax = next(it_axes)
        # z_hat = cdl.z_hat_[:, i_atom, :]
        z_hat = allZ[:, i_atom, :]
        if shift_acti:
            # roll to put activation to the peak amplitude time in the atom
            shift = np.argmax(np.abs(cdl.v_hat_[i_atom]))
            z_hat = np.roll(z_hat, shift, axis=1)
            z_hat[:, :shift] = 0  # pad with 0
        # t1 = np.arange(cdl.z_hat_.shape[2]) / sfreq - 1.7
        t1 = np.arange(allZ.shape[2]) / sfreq - activeStartTime
        ax.plot(t1, z_hat.T)
        ax.set_xlabel('Time (s)', fontsize=fontsize)

        if i_col == 0:
            ax.set_ylabel("Atom's activations", labelpad=7, fontsize=fontsize)

    plt.tight_layout()
    fig_name = 'global_figure' + figNameSuffix
    plt.savefig(subjectOutputDir / fig_name, dpi=300)
    plt.savefig(subjectOutputDir / (fig_name.replace('.pdf', '.png')), dpi=300)
    # bbox_inches='tight')
    plt.close()

    return None
