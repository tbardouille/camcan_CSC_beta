#!/usr/bin/env python

# %%
# Import libraries
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import mne
import multiprocessing as mp
import logging
from mne.time_frequency import tfr_morlet

# Script to calculate the TFR for each participant's MEG data


def MEG_preproc(subjectID):

    # Script to create processed data for Motor blocks
    # homeDir = Path(os.path.expanduser("~"))
    homeDir = Path.home()
    dataDir = homeDir / 'camcan'
    outDir = dataDir / 'proc_data' / 'TaskSensorAnalysis_transdef'
    dsPrefix = 'transdef_transrest_mf2pt2_task_raw'

    # Analysis parameters
    TFRfmin = 5
    TFRfmax = 50
    TFRfstep = 5
    channelName = 'MEG0712'

    ####################################################################
    # Files that exist
    subjectDir = outDir / str(subjectID)
    if not subjectDir.exists():
        subjectDir.mkdir(parents=True)
    fifName = dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo.fif'
    epochFif = subjectDir / fifName
    if not epochFif.exists():
        sys.exit("Put %s file into %s folder."
                 % (fifName, subjectDir))

    # Epoch data based on buttone press
    epochs_clean = mne.read_epochs(epochFif)

    # Isolate the channel of interest
    channelIndex = mne.pick_channels(
        epochs_clean.info["ch_names"], [channelName])

    # Calculate TFR on MEG data results
    logging.info('Starting TFR Calculation on Channel of Interest')
    epochs_no_evoked = epochs_clean.copy().subtract_evoked()
    freqs = np.arange(TFRfmin, TFRfmax, TFRfstep)
    n_cycles = freqs / 2.0
    power, _ = tfr_morlet(epochs_no_evoked, freqs=freqs, n_cycles=n_cycles,
                          picks=channelIndex, use_fft=False, return_itc=True,
                          decim=3, n_jobs=1)
    power.apply_baseline(baseline=(-1.5, -1))

    return power, epochs_clean


# Calculate the TFR for this participant at a sensor over left S1/M1
power, epochs_clean = MEG_preproc('CC620264')

# Plot the result, and make sure they have a nice beta suppression
power.plot()

# %%
