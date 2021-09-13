"""

"""
# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import mne

from dripp.trunc_norm_kernel.model import TruncNormKernel


def plot_csc(cdl_model, raw_csc, allZ,
             plot_acti_histo=False, shift_acti=True,
             activation_tstart=0, df_dripp=None,
             save_dir=Path('.'), title=None, show=True):
    """Plot the returns of CSC model.

    Parameters
    ----------
    cdl_model : instance of alphacsc.ConvolutionalDictionaryLearning

    raw_csc : mne.io.Raw
        The raw data on which CDL was run.

    allZ :

    plot_acti_histo : bool

    shift_acti : bool
        if True, roll to put activation to the peak amplitude time in the atom

    activation_tstart : float
        XXX I don't like you hard code 1.7 here.
        default is 0

    df_dripp : pandas.DataFrame
        possible DriPP results
        default is None

    save_dir : instance of pathlib.Path
        path to saving directory

    title : str
        title to put on figure

    show : bool
        Show figures at the end or not.

    Returns
    -------
    figs : list
        The list of generated matplotlib figures.
    """
    fontsize = 12
    n_atoms_per_fig = 5
    n_plot_per_atom = 3 + plot_acti_histo + (df_dripp is not None)
    n_atoms_est = allZ.shape[1]
    info = raw_csc.info
    sfreq = raw_csc.info['sfreq']
    atom_duration = cdl_model.v_hat_.shape[-1] / raw_csc.info['sfreq']
    figsize = (15, 10)

    atoms_in_figs = np.arange(0, n_atoms_est + 1, n_atoms_per_fig)
    atoms_in_figs = list(zip(atoms_in_figs[:-1], atoms_in_figs[1:]))

    figs = []
    for fig_idx, (atoms_start, atoms_stop) in enumerate(atoms_in_figs, start=1):
        fig, axes = plt.subplots(
            n_plot_per_atom, n_atoms_per_fig, figsize=figsize)
        figs.append(fig)
        fig.suptitle(title, fontsize=fontsize)
        for i_atom, kk in enumerate(range(atoms_start, atoms_stop)):
            ax = axes[0, i_atom]
            ax.set_title("Atom #" + str(kk), fontsize=fontsize)

            # Spatial pattern
            u_hat = cdl_model.u_hat_[kk]
            mne.viz.plot_topomap(u_hat, info, axes=ax, show=False)
            if i_atom == 0:
                ax.set_ylabel("Spatial", labelpad=86, fontsize=fontsize)

            # Temporal pattern
            ax = axes[1, i_atom]
            v_hat = cdl_model.v_hat_[kk]
            t = np.arange(v_hat.size) / sfreq
            ax.plot(t, v_hat)
            ax.grid(True)
            ax.set_xlim(0, atom_duration)  # crop x axis
            if i_atom == 0:
                ax.set_ylabel("Temporal", labelpad=14, fontsize=fontsize)

            # Power Spectral Density (PSD)
            ax = axes[2, i_atom]
            psd = np.abs(np.fft.rfft(v_hat, n=256)) ** 2
            frequencies = np.linspace(0, sfreq / 2.0, len(psd))
            ax.semilogy(frequencies, psd, label="PSD", color="k")
            ax.set_xlim(0, 40)  # crop x axis
            ax.set_xlabel("Frequencies (Hz)", fontsize=fontsize)
            ax.grid(True)
            if i_atom == 0:
                ax.set_ylabel("Power Spectral Density", labelpad=13,
                              fontsize=fontsize)

            if plot_acti_histo:
                # Atom's activations
                ax = axes[3, i_atom]
                z_hat = allZ[:, kk, :]
                if shift_acti:
                    # roll to put activation to the peak amplitude time
                    shift = np.argmax(np.abs(v_hat))
                    z_hat = np.roll(z_hat, shift, axis=1)
                    z_hat[:, :shift] = 0  # pad with 0
                t1 = np.arange(allZ.shape[2]) / sfreq - activation_tstart
                ax.plot(t1, z_hat.T)
                ax.set_xlabel("Time (s)", fontsize=fontsize)

                if i_atom == 0:
                    ax.set_ylabel("Atom's activations",
                                  labelpad=7, fontsize=fontsize)

            if (df_dripp is not None):
                # Atom's learned intensity
                ax = axes[4, i_atom]
                # get DriPP results
                columns = ['baseline', 'alpha', 'm', 'sigma', 'lower', 'upper']
                res = df_dripp[columns][df_dripp['atom'] == kk].iloc[0]
                baseline = res['baseline']
                lower, upper = res['lower'], res['upper']
                if np.isnan(baseline):
                    ax.text(0.5, 0.5, "no activation",
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=ax.transAxes)
                else:
                    xx = np.linspace(-0.5, upper, 500)
                    try:
                        # XXX
                        # for subject CC110033, atom 15, DriPP returns [nan]
                        # for alpha and kernel shape, but 0 for baseline
                        # define kernel function
                        alpha = res['alpha'][0]
                        m, sigma = res['m'][0], res['sigma'][0]
                        kernel = TruncNormKernel(lower, upper, m, sigma)
                        yy = baseline + alpha * kernel.eval(xx)
                    except AssertionError:
                        yy = baseline * np.ones(xx.shape)

                    # plot learned intensity
                    ax.plot(xx, yy, label='button')
                    ax.set_xlim(-0.5, upper)

                if i_atom == 0:
                    intensity_ax = ax
                    ax.set_ylabel("Intensity", labelpad=7, fontsize=fontsize)
                else:
                    # rescale inteisty axe
                    intensity_ax.get_shared_y_axes().join(intensity_ax, ax)
                    ax.autoscale()

                ax.legend(fontsize=fontsize, handlelength=1)

            fig.tight_layout()
            fig.subplots_adjust(top=0.88)

            fig_name = f"atoms_part_{fig_idx}.pdf"
            fig.savefig(save_dir / fig_name, dpi=300)
            fig.savefig(save_dir / (fig_name.replace(".pdf", ".png")),
                        dpi=300)

    if show:
        plt.show()
    return figs

# %%
