#!/usr/bin/env python

# %%

# import csv
import os
# from pathlib import Path
import sys
import numpy as np
# import pandas as pd
# import multiprocessing as mp
import logging

import mne
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs

from utils import getPaths, getGoodButtonsEvents


# Script to take raw MEG files from CamCAN and generate evoked response, with
# some other files along the way


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


def MEG_preproc(subjectID, apply_maxwell_filter=False):
    """

    Parameters
    ----------
    subjectID : str

    apply_maxwell_filter : bool
        if True, apply a Maxwell filter
        default is False (as Tim's file is pre-filtered)

    Returns
    -------
    0

    """

    # Epoching parameters
    prestim = -1.7
    poststim = 1.7
    baseStart = -1.25
    baseEnd = -1.0

    dsPrefix = 'transdef_transrest_mf2pt2_task_raw'

    # ========== Define all file paths ===========

    dictPaths = getPaths(subjectID)
    dataDir = dictPaths['dataDir']
    tsssFifDir = dictPaths['tsssFifDir']
    subjectOutDir = dictPaths['procSubjectOutDir']

    # Path to cross talk and calibration files for Maxwell filter
    ctSparseFile = dataDir / 'Cam-CAN_ct_sparse.fif'
    sssCalFile = dataDir / 'Cam-CAN_sss_cal.dat'

    if apply_maxwell_filter:
        tsssFifName = 'sub-' + str(subjectID) + '_ses-smt_task-smt_meg.fif'
    else:
        tsssFifName = dsPrefix + '.fif'

    tsssFif = tsssFifDir / tsssFifName
    if not tsssFif.exists():
        sys.exit("Put %s file into %s folder." % (tsssFifName, tsssFifDir))

    # Files that get made
    # subjectOutDir = outDir / str(subjectID)
    # if not subjectOutDir.exists():
    #     subjectOutDir.mkdir(parents=True)
    icaFif = subjectOutDir / (dsPrefix + '-ica.fif')
    # eveFif_all = subjectOutDir / (dsPrefix + '-eve.fif')
    # eveFif_button = subjectOutDir / \
    #     (dsPrefix + '_Under2SecResponseOnly-eve.fif')
    # file with epochs and evoked
    epochFif = subjectOutDir / \
        (dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo.fif')
    evokedFif = subjectOutDir / \
        (dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo-ave.fif')
    # file for cleaned raw data
    rawCleanFif = subjectOutDir / \
        (dsPrefix + '_cleaned.fif')

    ####################################################################

    # Setup log file for standarda output and error
    logFile = subjectOutDir / (dsPrefix + '_processing_notes.txt')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s',
        filename=logFile,
        filemode='a'
    )
    stdout_logger = logging.getLogger('STDOUT')
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl
    stderr_logger = logging.getLogger('STDERR')
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Read raw data
    raw = mne.io.read_raw_fif(tsssFif, preload=True)
    raw.filter(l_freq=None, h_freq=125)
    raw.notch_filter([50, 100])
    if apply_maxwell_filter:
        raw = mne.preprocessing.maxwell_filter(raw, calibration=sssCalFile,
                                               cross_talk=ctSparseFile,
                                               st_duration=10.0)

    evs, goodButtonEvents = getGoodButtonsEvents(
        raw, stim_channel='STI101', subtract_first_samp=True)

    # # Drop duplicate events in the button press list
    # evs_df = pd.DataFrame(goodButtonEvents)
    # goodButtonEvents = evs_df.drop_duplicates().values

    # # Now drop button presses that are within 3 seconds of the previous
    # # button press
    # goodButtonInterval = np.diff(
    #     goodButtonEvents[:, 0]) / raw.info['sfreq']
    # longDelay = np.where(goodButtonInterval > 3)[0]

    # # Write out event files for all stimuli/responses and "good" button
    # # presses only
    # mne.write_events(eveFif_all, evs)
    # mne.write_events(eveFif_button, goodButtonEvents)

    # Epoch data based on buttone press
    epochs = mne.Epochs(raw, events=np.array(goodButtonEvents), event_id=None,
                        tmin=prestim, tmax=poststim,
                        baseline=(baseStart, baseEnd),
                        verbose=False, preload=True)

    # Load or generate ICA decomposition for this dataset
    # performs ICA on data to remove artifacts according to rejection criteria
    if os.path.exists(icaFif):
        print('Reading ICA: %s' % icaFif)
        ica = mne.preprocessing.read_ica(icaFif)
    else:
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
                    str(subjectID)))

        # Reject bad ECG compoments following mne procedure
        ecg_epochs = create_ecg_epochs(raw, tmin=-0.5, tmax=0.5)
        ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
        ecg_inds = ecg_inds[:n_max_ecg]
        ica.exclude.extend(ecg_inds)
        # save ICA file
        ica.save(icaFif)

    # Apply ICA to epoched data	and save
    epochs_clean = epochs.copy()
    ica.apply(epochs_clean, exclude=ica.exclude)
    epochs_clean.save(epochFif, overwrite=True)
    # Average and save
    evoked = epochs_clean.average()
    evoked.save(evokedFif)
    # Print some info
    print("Subject ID: %s\nEpochs length: %i\nICA exclude: %s" %
          (str(subjectID), len(epochs), str(len(ica.exclude))))

    # Apply ICA to raw data	and save
    raw_clean = raw.copy()
    ica.apply(raw_clean, exclude=ica.exclude)
    raw_clean.save(rawCleanFif, overwrite=True)

    return 0


if __name__ == '__main__':

    # # Check number of subjects to be worked with
    # homeDir = Path(os.path.expanduser("~"))
    # goodSubjectsDir = homeDir / 'camcan' / 'proc_data'
    # if not goodSubjectsDir.exists():
    #     goodSubjectsDir.mkdir(parents=True)

    # goodSubjectsFile = goodSubjectsDir / 'demographics_goodSubjects.csv'
    # if not goodSubjectsFile.exists():
    #     sys.exit("Put demographics_goodSubjects.csv file into %s folder."
    #              % goodSubjectsDir)

    # goodSubjects = pd.read_csv(goodSubjectsFile)
    # goodSubjects = goodSubjects.loc[goodSubjects['DataReads'] == 1]
    # numSubjects = len(goodSubjects)
    # subjectIDs = goodSubjects['SubjectID'].tolist()

    # # Set up the parallel task pool to use all available processors
    # count = int(np.round(mp.cpu_count() * 3 / 4))
    # pool = mp.Pool(processes=count)

    # # Run the jobs
    # pool.map(MEG_preproc, subjectIDs)

    MEG_preproc('CC620264', apply_maxwell_filter=True)
    # MEG_preproc('CC620262')

# %%
