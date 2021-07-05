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

from alphacsc.viz.epoch import make_epochs


def plot_csc(subjectID, cdl_on_epoch=True, n_atoms=25, atomDuration=0.7,
             sfreq=150., sensorType='grad', reg=.2, prestim=-1.7,
             poststim=1.7, shift_acti=False):
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

    reg : float
        regularization parameter for the CSC
        default is 0.2

    prestim, poststim : float
        Epoching parameters when cdl_on_epoch is False

    shift_acti : bool
        if True, shift the atom's activation to put activation to the peak
        amplitude time in the atom
        default is False

    Returns
    -------
    no returns

    """

    # %%

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

    # fifName = dsPrefix + '_buttonPress_duration=3.4s_'
    # if cdl_on_epoch:
    #     fifName += 'cleaned-epo.fif'
    # else:
    #     fifName += 'cleaned-raw.fif'
    # megFile = inputDir / subjectID / fifName

    # Load in the CSC results
    if cdl_on_epoch:
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

    if cdl_on_epoch:
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
    if cdl_on_epoch:
        allZ = cdl.z_hat_
    else:
        # transform z_hat_ into epoched, with shape (n_events, n_atoms, duration)
        # file with good button events
        eveFif_button = inputDir / str(subjectID) / \
            (dsPrefix + '_Under2SecResponseOnly-eve.fif')
        goodButtonEvents = mne.read_events(eveFif_button)
        print("Number of events: %i" % len(goodButtonEvents))

        # allZ = np.array(z_hat_epo_)
        info['events'] = np.array(goodButtonEvents)
        info['event_id'] = None
        allZ = make_epochs(z_hat_, info,
                           t_lim=(prestim, poststim),
                           n_times_atom=int(np.round(atomDuration * sfreq)))

        print("allZ shape:", allZ.shape)
        # REMARK
        # allZ = array([...], shape=(6, 25, 511), dtype=float64)
        # epochs.drop_log = TOO SHORT?? -> Is that why we got no trial?
        # doc -> "'TOO_SHORT': If epoch didn't contain enough data names of
        # channels that exceeded the amplitude threshold"

        # events_tt = (goodButtonEvents[:, 0] / sfreq).astype(int)
        # z_hat_epo_ = []
        # for this_event_tt in events_tt:
        #     z_hat_trial = []
        #     for i_atom in range(n_atoms):
        #         z_hat = z_hat_[0][i_atom]
        #         # roll to put activation to the peak amplitude time in the atom
        #         shift = np.argmax(np.abs(cdl.v_hat_[i_atom]))
        #         z_hat = np.roll(z_hat, shift)
        #         z_hat[:shift] = 0  # pad with 0
        #         acti = z_hat[this_event_tt -
        #                      activeStart: this_event_tt + activeStart]
        #         z_hat_trial.append(acti)
        #     z_hat_epo_.append(z_hat_trial)

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
    plt.savefig(subjectOutputDir / ('global_figure' + figNameSuffix), dpi=300)
    # bbox_inches='tight')
    plt.close()

    # %%

    return None

# %%


if __name__ == '__main__':
    plot_csc(subjectID='CC620264', cdl_on_epoch=False, n_atoms=25,
             atomDuration=0.7, sfreq=150., reg=.2, shift_acti=True)
