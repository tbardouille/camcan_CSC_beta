# %%
import numpy as np
import matplotlib.pyplot as plt
import mne

from utils import getPaths, getGoodButtonsEvents


# audiovis/300Hz: 1
# audiovis/600Hz: 2
# audiovis/1200Hz: 3
# catch/0: 4
# catch/1: 5
# button: 8192

use_drago = True
cdl_on_epoch = False
sensorType = 'grad'
sfreq = 150.
subjectID = 'CC620264'  # age: 76.33
# subjectID = 'CC110037'  # age: 18.75
# subjectID = 'CC723395'  # age: 86.08
prestim, poststim = -1.7, 1.7
baseStart, baseEnd = -1.25, -1.0


dictPaths = getPaths(subjectID, use_drago=use_drago)
subjectInputDir = dictPaths['procSubjectOutDir']
subjectOutputDir = dictPaths['cscSubjectOutDir']
subjectOutDir = dictPaths['procSubjectOutDir']

dsPrefix = 'transdef_transrest_mf2pt2_task_raw'

if cdl_on_epoch:
    fifName = dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo.fif'
else:
    fifName = dsPrefix + '_cleaned.fif'
megFile = subjectInputDir / fifName

raw = mne.io.read_raw_fif(megFile, preload=True)
raw.pick_types(meg=sensorType, stim=True)

# Band-pass filter the data to a range of interest
raw.filter(l_freq=2, h_freq=45)

events = mne.find_events(raw, 'STI101', shortest_event=1)
raw, events = raw.resample(
    sfreq, npad='auto', verbose=False, events=events)

_, goodButtonEvents = getGoodButtonsEvents(
    raw, stim_channel='STI101', subtract_first_samp=False)

# epochs_events = np.concatenate((events, goodButtonEvents))
event_dict = {'audiovis/300Hz': 1, 'audiovis/600Hz': 2, 'audiovis/1200Hz': 3,
              'catch/0': 4, 'catch/1': 5, 'button': 8192}  # , 'good_button': 128}
# Epoch data based on buttone press
epochs = mne.Epochs(raw, events=events, event_id=event_dict,
                    tmin=prestim, tmax=poststim,
                    baseline=(baseStart, baseEnd),
                    verbose=False, preload=True)
# only auditory/visual events
evoked_audiovis = epochs['audiovis'].average()
fig = evoked_audiovis.plot_joint()
fig.savefig('evoked_audiovis.png')
fig.savefig('evoked_audiovis.pdf')
fig.clear()
# all button events
evoked_button = epochs['button'].average()
fig = evoked_button.plot_joint()
fig.savefig('evoked_button.png')
fig.savefig('evoked_button.pdf')
fig.clear()
# only "good" button events
epochs = mne.Epochs(raw, events=goodButtonEvents, event_id=None,
                    tmin=prestim, tmax=poststim,
                    baseline=(baseStart, baseEnd),
                    verbose=False, preload=True)
evoked_good_button = epochs.average()
fig = evoked_good_button.plot_joint()
fig.savefig('evoked_good_button.png')
fig.savefig('evoked_good_button.pdf')
fig.clear()

# evokedFif = subjectOutDir / \
#     (dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo-ave.fif')
# evoked = mne.read_evokeds(evokedFif, baseline=(baseStart, baseEnd))
# evoked[0].plot_joint()

# %%
