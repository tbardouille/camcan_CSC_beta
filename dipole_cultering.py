"""
Compute a non-supervised clustering based on the 3D position of dipole fit for
each atom of every subject computed using CDL.
"""
# %%
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from joblib import Parallel, delayed
from sklearn import cluster
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt

import mne

DATA_DIR = Path("/storage/store/data/")
BEM_DIR = DATA_DIR / "camcan-mne/freesurfer"
TRANS_DIR = DATA_DIR / "camcan-mne/trans"

RESULTS_DIR = Path('./results_csc')
SUBJECT_DIRS = [f for f in RESULTS_DIR.iterdir() if not f.is_file()]

N_JOBS = 20

# %%


def compute_dipole_fit(subject_dir):
    """
    For a given subject, compute the dipole fit of each of the pre-computed
    atoms, and save the 3D coordinates in a pandas DataFrame.

    Parameters
    ----------
    subject_dir : PosixPath
        the path to the CDL results of the given subject id

    Returns
    -------
    None
    """
    # get the files needed for dipole fitting
    subject_id = subject_dir.name
    fname_bem = BEM_DIR / subject_id / 'bem' / (subject_id + '-meg-bem.fif')
    trans = TRANS_DIR / ('sub-' + subject_id + '-trans.fif')

    if fname_bem.exists() and trans.exists():
        if not (subject_dir / 'df_dip_pos.csv').exists():
            # get CDL results
            file_name = subject_dir / 'CSCraw_0.5s_20atoms.pkl'
            cdl_model, info, _, _ = pickle.load(open(file_name, "rb"))
            # compute noise covariance
            cov = mne.make_ad_hoc_cov(info)
            u_hat_ = cdl_model.u_hat_
            # select only grad channels
            meg_indices = mne.pick_types(info, meg='grad')
            info = mne.pick_info(info, meg_indices)
            evoked = mne.EvokedArray(u_hat_.T, info)
            # compute dipole fit
            dip = mne.fit_dipole(evoked, cov, str(fname_bem), str(trans),
                                 n_jobs=6, verbose=False)[0]
            df_dip_pos = pd.DataFrame(data=dip.pos, columns=['x', 'y', 'z'])
            df_dip_pos['atom_id'] = range(dip.pos.shape[0])
            df_dip_pos['subject_id'] = subject_id
            df_dip_pos.to_csv(subject_dir / 'df_dip_pos.csv', index=False)
    else:
        print("subject_id:", subject_id)
        print('fname_bem: %s, trans: %s' %
              (fname_bem.exists(), trans.exists()))


Parallel(n_jobs=N_JOBS, verbose=1)(
    delayed(compute_dipole_fit)(subject_dir) for subject_dir in SUBJECT_DIRS)

# Merge df_dip_pos accross subjects
df_dip_pos_all = pd.DataFrame()
for subject_dir in SUBJECT_DIRS:
    if (subject_dir / 'df_dip_pos.csv').exists():
        df_dip_pos = pd.read_csv(subject_dir / 'df_dip_pos.csv')
        df_dip_pos_all = df_dip_pos_all.append(df_dip_pos, ignore_index=True)

df_dip_pos_all.to_csv(RESULTS_DIR / 'df_dip_pos_all.csv', index=False)


# %% Compute CAH dendograme
df = pd.read_csv(RESULTS_DIR / 'df_dip_pos_all.csv')
columns = ['x', 'y', 'z']
data = np.array(df[columns])

plt.title("CAH on CAM-Can dipole fit")
dendrogram(linkage(data, method='ward'), labels=df.index,
           orientation='right', truncate_mode='level', p=5)
plt.savefig(RESULTS_DIR / 'dendrogram.pdf')
plt.show()


# %% Select number of classes to keep
n_clusters = 5

# Recompute CAH clustering
clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                     affinity='euclidean', linkage='ward')
clustering.fit(data)
labels_cah = clustering.labels_
df['labels_cah'] = labels_cah

# With k-means
kmeans = cluster.KMeans(n_clusters=n_clusters)
kmeans.fit(data)
labels_kmeans = kmeans.labels_
df['labels_kmeans'] = labels_kmeans


def plot_topomaps(subject_id, df, save_path=None):
    """
    For a given subject id and a dataframe of clustering results for a given
    class, plot the atoms' topomaps.

    Parameters
    ----------
    subject_id : str

    df : pandas DataFrame
        the clustering results of one class

    save_path : PosixPath | None
        the path to save the obtained figure


    Returns
    -------
    None
    """
    # get the atom idexes to plot
    atom_idx = df[df['subject_id'] == subject_id]['atom_id'].values
    # get CDL results for the subject_id
    file_name = RESULTS_DIR / subject_id / 'CSCraw_0.5s_20atoms.pkl'
    cdl_model, info, _, _ = pickle.load(open(file_name, "rb"))
    u_hat_ = cdl_model.u_hat_
    # select only grad channels in Info
    meg_indices = mne.pick_types(info, meg='grad')
    info = mne.pick_info(info, meg_indices)
    # plot topomaps
    n_columns = min(4, len(atom_idx))
    split = int(np.ceil(len(atom_idx) / n_columns))
    figsize = (4 * n_columns, 3 * split)
    fig, axes = plt.subplots(split, n_columns, figsize=figsize)
    axes = np.atleast_2d(axes)
    for ii, kk in enumerate(atom_idx):
        i_row, i_col = ii // n_columns, ii % n_columns
        it_axes = iter(axes[i_row:(i_row + 1), i_col])

        ax = next(it_axes)
        ax.set_title('Atom % d Subject %s' % (kk, subject_id), pad=0)
        mne.viz.plot_topomap(data=u_hat_[kk], pos=info, axes=ax, show=False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path / (subject_id + '.pdf'))
    else:
        plt.show()
    plt.close()


for method in ['kmeans', 'cah']:
    # create saving directory
    results_clustering = RESULTS_DIR / method
    results_clustering.mkdir(parents=True, exist_ok=True)

    if method == 'kmeans':
        labels = 'labels_kmeans'
    elif method == 'cah':
        labels = 'labels_cah'

    for this_label in set(df[labels].values):
        print('Label %i' % this_label)
        # create saving sub-directory
        save_path = results_clustering / str(this_label)
        save_path.mkdir(parents=True, exist_ok=True)
        # select sub_df
        sub_df = df[df[labels] == this_label]
        # for every subject_id, plot the topomaps in the class
        Parallel(n_jobs=N_JOBS, verbose=1)(
            delayed(plot_topomaps)(subject_id, sub_df, save_path)
            for subject_id in set(sub_df['subject_id'].values))

# %%
