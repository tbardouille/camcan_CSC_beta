"""

"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import mne


def plot_csc(n_atoms_est, atom_duration, cdl_model, info, sfreq=150.,
             plot_acti_histo=False, allZ=[], shift_acti=True,
             activation_tstart=1.7, save_dir=Path('.')):
    """

    Parameters
    ----------
    n_atoms_est : int

    atom_duration : 

    cdl_model : 

    info

    sfreq

    plot_acti_histo : bool

    allZ : 

    shift_acti : bool


    save_dir : instance of pathlib.Path
        path to saving directory

    """
    fontsize = 12
    n_atoms_per_fig = 5
    n_plot_per_atom = 3 + plot_acti_histo
    figsize = (15, 7)

    atoms_in_figs = np.arange(0, n_atoms_est + 1, n_atoms_per_fig)
    atoms_in_figs = list(zip(atoms_in_figs[:-1], atoms_in_figs[1:]))

    for fig_idx, (atoms_start, atoms_stop) in enumerate(atoms_in_figs, start=1):
        fig, axes = plt.subplots(
            n_plot_per_atom, n_atoms_per_fig, figsize=figsize)

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
                z_hat = allZ[:, i_atom, :]
                if shift_acti:
                    # roll to put activation to the peak amplitude time in the atom
                    shift = np.argmax(np.abs(cdl_model.v_hat_[i_atom]))
                    z_hat = np.roll(z_hat, shift, axis=1)
                    z_hat[:, :shift] = 0  # pad with 0
                # t1 = np.arange(cdl.z_hat_.shape[2]) / sfreq - 1.7
                t1 = np.arange(allZ.shape[2]) / sfreq - activation_tstart
                ax.plot(t1, z_hat.T)
                ax.set_xlabel("Time (s)", fontsize=fontsize)

                if i_atom == 0:
                    ax.set_ylabel("Atom's activations",
                                  labelpad=7, fontsize=fontsize)

            fig.tight_layout()

            fig_name = f"atoms_part_{fig_idx}.pdf"
            fig.savefig(save_dir / fig_name, dpi=300)
            fig.savefig(save_dir / (fig_name.replace(".pdf", ".png")),
                        dpi=300)
            # fig.close()

    # plt.show()
