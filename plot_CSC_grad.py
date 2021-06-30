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


def plot_csc(subjectID, use_epoch=True, n_atoms=25, atomDuration=0.7,
             sfreq=150., sensorType='grad', reg=.2, prestim=-1.7,
             poststim=1.7):
    """

    Parameters
    ----------
    subjectID : str
        Subject to analyse

    use_epoch : bool
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

    reg : float
        regularization parameter for the CSC
        default is 0.2

    prestim, poststim : float
        Epoching parameters when use_epoch is False

    Returns
    -------
    no returns

    """

    # Paths
    # homeDir = Path(os.path.expanduser("~"))
    homeDir = Path.home()
    inputDir = homeDir / 'camcan' / \
        'proc_data' / 'TaskSensorAnalysis_transdef'
    outputDir = homeDir / 'data' / 'CSC'

    # [seconds from start of trial, i.e., -1.7 seconds wrt cue]
    preStimStartTime = 0
    activeStartTime = 1.7   # [seconds from start of trial
    zWindowDurationTime = 0.5   # in seconds

    subjectOutputDir = outputDir / subjectID
    dsPrefix = 'transdef_transrest_mf2pt2_task_raw'

    # Load in the CSC results
    fifName = dsPrefix + '_buttonPress_duration=3.4s_'
    if use_epoch:
        fifName += 'cleaned-epo.fif'
    else:
        fifName += 'cleaned-raw.fif'
    megFile = inputDir / subjectID / fifName

    if use_epoch:
        pkl_name = 'CSCepochs_'
    else:
        pkl_name = 'CSCraw_'
    pkl_name += str(int(atomDuration * 1000)) + \
        'ms_' + sensorType + str(n_atoms) + 'atoms_' + \
        str(reg * 10) + 'reg.pkl'
    outputFile = subjectOutputDir / pkl_name

    res = pickle.load(open(outputFile, "rb"))
    if len(res) == 2:
        cdl, info = res
        z_hat_ = cdl.z_hat_
    elif len(res) == 3:
        cdl, info, z_hat_ = res

    sfreq = info['sfreq']

    preStimStart = int(np.round(preStimStartTime * sfreq))
    activeStart = int(np.round(activeStartTime * sfreq))
    zWindowDuration = int(np.round(zWindowDurationTime * sfreq))

    if use_epoch:
        figNameSuffix = '_CSCepo'
    else:
        figNameSuffix = '_CSCraw'
    figNameSuffix += '_' + str(int(atomDuration * 1000)) + \
        'ms_' + sensorType + str(n_atoms) + 'atoms_' + \
        str(reg * 10) + 'reg.pdf'

    # # Plot time course for all atoms
    # fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(11, 8.5))
    # for i_atom in range(n_atoms):
    #     ax = np.ravel(axes)[i_atom]
    #     v_hat = cdl.v_hat_[i_atom]
    #     t = np.arange(v_hat.size) / sfreq
    #     ax.plot(t, v_hat)
    #     ax.set(title='Atom #' + str(i_atom))
    #     ax.grid(True)
    # plt.savefig(subjectOutputDir / 'time_course.pdf', dpi=300)
    # plt.show()

    # # Plotting sensor weights from U vectors as topographies
    # fig, axes = plt.subplots(5, 5, figsize=(11, 8.5))
    # for i_atom in range(n_atoms):
    #     ax = np.ravel(axes)[i_atom]
    #     u_hat = cdl.u_hat_[i_atom]
    #     mne.viz.plot_topomap(u_hat, info, axes=ax, show=False)
    #     ax.set(title='Atom #' + str(i_atom))
    # plt.savefig(subjectOutputDir / 'sensor_weights.pdf', dpi=300)
    # plt.show()

    # # Plot z vector overlaid for all trials (for each atom)
    # # first, make a time vector for the x-axis
    # t1 = np.arange(cdl.z_hat_.shape[2]) / sfreq - 1.7
    # # Then, plot each atom's Z
    # fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(11, 8.5))
    # for i_atom in range(n_atoms):
    #     ax = np.ravel(axes)[i_atom]
    #     z_hat = cdl.z_hat_[:, i_atom, :]
    #     ax.plot(t1, z_hat.T)
    #     ax.set(title='Atom #' + str(i_atom))
    # plt.savefig(subjectOutputDir / 'atoms_z.pdf', dpi=300)
    # plt.show()

    # get effective n_atoms
    n_atoms = z_hat_.shape[1]
    # transform z_hat_ into epoched, with shape (n_events, n_atoms, duration)
    if not use_epoch:
        # file with good button events
        eveFif_button = inputDir / str(subjectID) / \
            (dsPrefix + '_Under2SecResponseOnly-eve.fif')
        goodButtonEvents = mne.read_events(eveFif_button)
        events_tt = (goodButtonEvents[:, 0] / sfreq).astype(int)

        z_hat_epo_ = []
        for this_event_tt in events_tt:
            z_hat_trial = []
            for i_atom in range(n_atoms):
                z_hat = z_hat_[0][i_atom]
                # roll to put activation to the peak amplitude time in the atom
                shift = np.argmax(np.abs(cdl.v_hat_[i_atom]))
                z_hat = np.roll(z_hat, shift)
                z_hat[:shift] = 0  # pad with 0
                acti = z_hat[this_event_tt -
                             activeStart: this_event_tt + activeStart]
                z_hat_trial.append(acti)
            z_hat_epo_.append(z_hat_trial)

        allZ = np.array(z_hat_epo_)
    else:
        allZ = cdl.z_hat_

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

    # Plot general figure: spatial & temporal pattern, power spectral density (PSD)
    # and activations

    # if not use_epoch:
    #     # file with epochs
    #     # epochFif = subjectOutputDir / \
    #     #     (dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo.fif')
    #     # epochs_clean = mne.read_epochs(epochFif)
    #     # events_tt = (epochs_clean.events[:, 0] / sfreq).astype(int)
    #     # tmin, tmax = epochs_clean.tmin, epochs_clean.tmax
    #     # ttleft, ttright = int(abs(tmin) * sfreq), int(abs(tmax) * sfreq)

    #     # file with good button events
    #     eveFif_button = inputDir / str(subjectID) / \
    #         (dsPrefix + '_Under2SecResponseOnly-eve.fif')
    #     goodButtonEvents = mne.read_events(eveFif_button)
    #     events_tt = (goodButtonEvents[:, 0] / sfreq).astype(int)
    #     ttleft, ttright = int(abs(prestim) * sfreq), int(abs(poststim) * sfreq)

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
            ax.set_ylabel('Spatial', labelpad=70, fontsize=20)

        # Temporal pattern
        ax = next(it_axes)
        v_hat = cdl.v_hat_[i_atom]
        t = np.arange(v_hat.size) / sfreq
        ax.plot(t, v_hat)
        ax.grid(True)
        ax.set_xlim(0, atomDuration)  # crop x axis
        if i_col == 0:
            ax.set_ylabel('Temporal', labelpad=14, fontsize=20)

        # Power Spectral Density (PSD)
        ax = next(it_axes)
        psd = np.abs(np.fft.rfft(v_hat, n=256)) ** 2
        frequencies = np.linspace(0, sfreq / 2.0, len(psd))
        ax.semilogy(frequencies, psd, label='PSD', color='k')
        ax.set_xlim(0, 40)  # crop x axis
        ax.set(xlabel='Frequencies (Hz)', fontsize=20)
        ax.grid(True)
        if i_col == 0:
            ax.set_ylabel('Power Spectral Density', labelpad=13, fontsize=20)

        # Atom's activations
        ax = next(it_axes)
        # z_hat = cdl.z_hat_[:, i_atom, :]
        # t1 = np.arange(cdl.z_hat_.shape[2]) / sfreq - 1.7
        z_hat = allZ[:, i_atom, :]
        t1 = np.arange(allZ.shape[2]) / sfreq - activeStartTime
        ax.plot(t1, z_hat.T)
        ax.set(xlabel='Time (s)', fontsize=20)

        if i_col == 0:
            ax.set_ylabel("Atom's activations", labelpad=20, fontsize=20)

    plt.tight_layout()
    plt.savefig(subjectOutputDir / ('global_figure' + figNameSuffix), dpi=300)
    # bbox_inches='tight')
    plt.close()

    return None


if __name__ == '__main__':
    plot_csc(subjectID='CC620264', use_epoch=False, n_atoms=30,
             atomDuration=0.7, sfreq=150., reg=.1)
