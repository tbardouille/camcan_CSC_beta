"""
Utils functions used accross all scripts
"""

import numpy as np
import pandas as pd
from pathlib import Path

import mne

# Define the cache directory for joblib.Memory
CACHEDIR = Path('./__cache__')


def getPaths(subjectID):
    """

    Parameters
    ----------
    subjectID : string
        the subject ID we are interested in

    Returns
    -------
    dictPaths : dict
        dictionary of the Paths of interest for the given subject ID

    """

    # initialize Paths dictionary
    dictPaths = {}

    homeDir = Path.home()
    dataDir = homeDir / 'camcan'
    dictPaths['dataDir'] = dataDir
    # path to raw data
    tsssFifDir = dataDir / 'megData_moveComp' / str(subjectID) / 'task'
    dictPaths['tsssFifDir'] = tsssFifDir
    # path to save pre-processing data
    procSubjectOutDir = dataDir / 'proc_data' / \
        'TaskSensorAnalysis_transdef' / str(subjectID)
    dictPaths['procSubjectOutDir'] = procSubjectOutDir
    # path to save CSC results and final figures
    cscSubjectOutDir = homeDir / 'data' / 'CSC' / str(subjectID)
    dictPaths['cscSubjectOutDir'] = cscSubjectOutDir

    # behaviouralDir = dataDir / 'behaviouralData'
    # demographicFile = dataDir / 'proc_data' / 'demographics_goodSubjects.csv'

    for pathDir in dictPaths.values():
        # create directory if not already exists
        pathDir.mkdir(parents=True, exist_ok=True)

    return dictPaths


def getGoodButtonsEvents(raw, stim_channel='STI101', subtract_first_samp=True):
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
    evs : array-like of shape (n_events, 3)
        all events

    goodButtonEvents : array-like of shape (n_ButtonEvents, 3)
        only "good" button events
        the ID (third column) is set to 128 for all events

    """
    # Find button presses to stimuli
    evs = mne.find_events(raw, stim_channel, shortest_event=1)
    if subtract_first_samp:
        evs[:, 0] -= raw.first_samp
    # Get stimuli and response latencies
    # Pull event IDs
    evtID = evs[:, 2]
    # Get all events with ID < 10 (cues)
    stimEvents = evs[np.where(evtID < 10)[0], :]
    stimOnsets = stimEvents[:, 0]
    # Get all events with ID > 10 (button press) - not always the same number
    buttonEvents = evs[np.where(evtID > 10)[0], :]
    # Make the button press event always have ID = 128
    buttonOnsets = buttonEvents[:, 0]
    buttonEvents[:, 2] = 128
    # Stimulus loop to find the next button press under 2 seconds
    goodButtonEvents = []
    # Loop per cue
    for thisStimSample in stimOnsets:
        # Find timing of responses wrt stimulus
        allRTs = buttonOnsets - thisStimSample
        # Find where this timing is positive
        positiveRTIndex = np.where(allRTs > 0)[0]
        # If there is a positive response timing ...
        if len(positiveRTIndex) > 0:
            # And if that positive timing is less than 1 second ...
            thisRT = allRTs[positiveRTIndex][0] / raw.info['sfreq']
            if thisRT < 1:
                # Then also check that this is the first button press, or the
                # previous response was more than 3 seconds ago
                thisButtonPressEvent = buttonEvents[positiveRTIndex[0], :]
                thisOnset = thisButtonPressEvent[0]
                relativeButtonSamples = buttonOnsets - thisOnset
                priorBPIndex = np.where(relativeButtonSamples < 0)[0]
                # If this is the first button press
                if len(priorBPIndex) == 0:
                    if len(goodButtonEvents) == 0:
                        goodButtonEvents = thisButtonPressEvent
                    else:
                        goodButtonEvents = np.vstack(
                            (goodButtonEvents, thisButtonPressEvent))
                else:
                    # If not, check the time from previous response
                    samplesToPriorResponse = \
                        relativeButtonSamples[priorBPIndex[-1]]
                    timeToPriorResponse = -1 * \
                        samplesToPriorResponse / raw.info['sfreq']
                    if timeToPriorResponse > 3:
                        # Then either start a matrix with good button press
                        # events, or add to it
                        if len(goodButtonEvents) == 0:
                            goodButtonEvents = thisButtonPressEvent
                        else:
                            goodButtonEvents = np.vstack(
                                (goodButtonEvents, thisButtonPressEvent))

    if len(goodButtonEvents) == 0:
        print("No 'good' button event found.")
    else:
        # Drop duplicate events in the button press list
        evs_df = pd.DataFrame(goodButtonEvents)
        goodButtonEvents = evs_df.drop_duplicates().values

    return evs, goodButtonEvents


def getCSCPickleName(use_batch_cdl=True, use_greedy_cdl=False,
                     cdl_on_epoch=False, atomDuration=0.7, sensorType='grad',
                     n_atoms=25, reg=.2, eps=1e-4, tol_z=1e-2):
    """

    """
    assert (use_batch_cdl + use_greedy_cdl) == 1

    if use_batch_cdl:
        pkl_name = 'Batch'
    elif use_greedy_cdl:
        pkl_name = 'Greedy'

    if cdl_on_epoch:
        pkl_name += 'CSCepochs_'
    else:
        pkl_name += 'CSCraw_'
    pkl_name += str(int(atomDuration * 1000)) + \
        'ms_' + sensorType + str(n_atoms) + 'atoms_' + \
        str(reg) + 'reg' + str(eps) + 'eps' + str(tol_z) + 'tol_z'

    pkl_name = pkl_name.replace('.', '')
    pkl_name += '.pkl'

    return pkl_name
