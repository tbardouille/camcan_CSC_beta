# %%
import os
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import scipy.stats as ss
from scipy.signal import tukey
import multiprocessing as mp
import matplotlib.pyplot as plt

import mne

# if True, use the epoched data file, if False, use ful length data
make_epoch = False

# Paths
# homeDir = Path(os.path.expanduser("~"))
homeDir = Path.home()
inputDir = homeDir / 'data' / 'camcan' / \
    ' proc_data' / 'TaskSensorAnalysis_transdef'
outputDir = homeDir / 'data/CSC'

n_atoms = 25
subjectID = 'CC620264'
sensorType = 'grad'
atomDuration = 0.5  # in seconds

# [seconds from start of trial, i.e., -1.7 seconds wrt cue]
preStimStartTime = 0
activeStartTime = 1.7   # [seconds from start of trial
zWindowDurationTime = 0.5   # in seconds

subjectOutputDir = outputDir / subjectID

# Load in the CSC results
fifName = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_'
if make_epoch:
    fifName += 'cleaned-epo.fif'
else:
    fifName += 'cleaned-raw.fif'
megFile = inputDir / subjectID / fifName


if make_epoch:
    pkl_name = 'CSCepochs_'
else:
    pkl_name = 'CSCraw_'
pkl_name += str(int(atomDuration*1000)) + \
    'ms_' + sensorType + str(n_atoms) + 'atoms.pkl'
outputFile = subjectOutputDir / pkl_name

cdl, info = pickle.load(open(outputFile, "rb"))

sfreq = info['sfreq']

preStimStart = int(np.round(preStimStartTime*sfreq))
activeStart = int(np.round(activeStartTime*sfreq))
zWindowDuration = int(np.round(zWindowDurationTime*sfreq))

# %%

# Plot time course for all atoms
fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(11, 8.5))
for i_atom in range(n_atoms):
    ax = np.ravel(axes)[i_atom]
    v_hat = cdl.v_hat_[i_atom]
    t = np.arange(v_hat.size) / sfreq
    ax.plot(t, v_hat)
    ax.set(title='Atom #' + str(i_atom))
    ax.grid(True)
plt.savefig(subjectOutputDir / 'time_course.pdf', dpi=300)
plt.show()


# Plotting sensor weights from U vectors as topographies
fig, axes = plt.subplots(5, 5, figsize=(11, 8.5))
for i_atom in range(n_atoms):
    ax = np.ravel(axes)[i_atom]
    u_hat = cdl.u_hat_[i_atom]
    mne.viz.plot_topomap(u_hat, info, axes=ax, show=False)
    ax.set(title='Atom #' + str(i_atom))
plt.savefig(subjectOutputDir / 'sensor_weights.pdf', dpi=300)
plt.show()


# Plot z vector overlaid for all trials (for each atom)
# first, make a time vector for the x-axis
t1 = np.arange(cdl.z_hat_.shape[2]) / sfreq - 1.7
# Then, plot each atom's Z
fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(11, 8.5))
for i_atom in range(n_atoms):
    ax = np.ravel(axes)[i_atom]
    z_hat = cdl.z_hat_[:, i_atom, :]
    ax.plot(t1, z_hat.T)
    ax.set(title='Atom #' + str(i_atom))
plt.savefig(subjectOutputDir / 'atoms_z.pdf', dpi=300)
plt.show()

# Calculate the sum of all Z values for prestim and active intervals
allZ = cdl.z_hat_
activezHat = allZ[:, :, activeStart:activeStart+zWindowDuration]
preStimzHat = allZ[:, :, preStimStart:preStimStart+zWindowDuration]
activeSum = np.sum(np.sum(activezHat, axis=2), 0)
preStimSum = np.sum(np.sum(preStimzHat, axis=2), 0)
diffZ = activeSum-preStimSum

# Plot the change in activation
plt.figure()
plt.bar(x=range(25), height=diffZ)
plt.ylabel('Aggregate Activation Change')
plt.xlabel('Atom #')
plt.savefig(subjectOutputDir / 'change_in_activation.pdf', dpi=300)
plt.show()


# T-test of z-value sums across trials between intervals
activeSumPerTrial = np.sum(activezHat, axis=2)
preStimSumPerTrial = np.sum(preStimzHat, axis=2)
diffZperTrial = activeSumPerTrial-preStimSumPerTrial
tstat, pvalue = ss.ttest_1samp(diffZperTrial, popmean=0, axis=0)

# Plot result of the t-test
plt.figure()
plt.bar(x=range(25), height=tstat)
plt.grid(True)
plt.ylabel('t-stat')
plt.xlabel('Atom #')
plt.savefig(subjectOutputDir / 't_test.pdf', dpi=300)
plt.show()

# Plot general figure: spatial & temporal pattern, power spectral density (PSD)
# and activations

n_plots = 4
n_columns = min(5, n_atoms)
split = int(np.ceil(n_atoms / n_columns))
figsize = (4 * n_columns, 3 * n_plots * split)
fig, axes = plt.subplots(n_plots * split, n_columns, figsize=figsize)

for i_atom, kk in enumerate(range(n_atoms)):
    i_row, i_col = i_atom // n_columns, i_atom % n_columns
    it_axes = iter(axes[i_row * n_plots:(i_row + 1) * n_plots, i_col])

    ax = next(it_axes)
    ax.set(title='Atom #' + str(i_atom))

    # Spatial pattern
    u_hat = cdl.u_hat_[i_atom]
    mne.viz.plot_topomap(u_hat, info, axes=ax, show=False)
    if i_col == 0:
        ax.set_ylabel('Spatial', labelpad=40)

    # Temporal pattern
    ax = next(it_axes)
    v_hat = cdl.v_hat_[i_atom]
    t = np.arange(v_hat.size) / sfreq
    ax.plot(t, v_hat)
    ax.grid(True)
    if i_col == 0:
        ax.set_ylabel('Temporal', labelpad=20)

    # Power Spectral Density (PSD)
    ax = next(it_axes)
    psd = np.abs(np.fft.rfft(v_hat, n=256)) ** 2
    frequencies = np.linspace(0, sfreq / 2.0, len(psd))
    ax.semilogy(frequencies, psd, label='PSD', color='k')
    ax.set(xlabel='Frequencies (Hz)')
    ax.grid(True)
    if i_col == 0:
        ax.set_ylabel('Power Spectral Density', labelpad=20)

    # Atom's activations
    ax = next(it_axes)
    z_hat = cdl.z_hat_[:, i_atom, :]
    ax.plot(t1, z_hat.T)
    ax.set(xlabel='Time (s)')
    if i_col == 0:
        ax.set_ylabel("Atom's activations", labelpad=20)

plt.savefig(subjectOutputDir / 'global_figure.pdf', dpi=300)
