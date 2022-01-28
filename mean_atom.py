# %%
from alphacsc.utils.convolution import construct_X, construct_X_multi
from alphacsc.utils.dictionary import get_D
from alphacsc import BatchCDL, GreedyCDL
import alphacsc
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from sklearn import cluster
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt

from joblib import Memory, Parallel, delayed

import mne

from utils_csc import run_csc

# Paths for Cam-CAN dataset
DATA_DIR = Path("/storage/store/data/")
BEM_DIR = DATA_DIR / "camcan-mne/freesurfer"
TRANS_DIR = DATA_DIR / "camcan-mne/trans"
BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"  # Root path to raw BIDS files
PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"

# path to all participant CSC results
RESULTS_DIR = Path('./results_csc')
SUBJECT_DIRS = [f for f in RESULTS_DIR.iterdir() if not f.is_file()]

AGE_GROUP = 1


def get_df_topomaps(age_group):
    """

    """
    # get the subject's folders that correspond to the age group
    age_group_dir = [subject_dir for subject_dir in SUBJECT_DIRS
                     if subject_dir.name[:3] == ('CC' + str(age_group))]

    # For every subject in the age group, get the associated topomap vector
    age_group_u = {this_dir.name: pickle.load(
        open(this_dir / 'CSCraw_0.5s_20atoms.pkl', "rb"))[0].u_hat_
        for this_dir in age_group_dir}

    # Create a dataframe containing all topomaps
    df_topomaps = pd.DataFrame()
    for subject_id, u_hat in age_group_u.items():
        for i, u in enumerate(u_hat):
            new_row = {i: u[i] for i in range(len(u))}
            new_row['subject_id'] = subject_id
            new_row['atom_id'] = int(i)
            df_topomaps = df_topomaps.append(new_row, ignore_index=True)

    data_columns = [i for i in range(len(u))]

    return df_topomaps, data_columns


def df_culstering(df, data_columns, n_clusters):
    """

    """
    data = np.array(df[data_columns])
    # Recompute CAH clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                         affinity='euclidean',
                                         linkage='ward')
    clustering.fit(data)
    df['labels_cah'] = clustering.labels_

    # With k-means
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    df['labels_kmeans'] = kmeans.labels_

    return df


def plot_topomaps(subject_id, atom_idx):
    # get CDL results for the subject_id
    file_name = RESULTS_DIR / subject_id / 'CSCraw_0.5s_20atoms.pkl'
    cdl_model, info, _, _ = pickle.load(open(file_name, "rb"))
    u_hat_ = cdl_model.u_hat_
    # select only grad channels
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
    plt.show()


def reconstruct_class_signal(df, method, label):
    """

    """

    sub_df = df[df['labels_' + method] == label]

    Z_temp = []
    D_temp = []
    min_n_times_valid = np.inf
    for subject_id in list(set(sub_df['subject_id'].values)):
        file_name = RESULTS_DIR / subject_id / 'CSCraw_0.5s_20atoms.pkl'
        cdl_model, info, _, _ = pickle.load(open(file_name, "rb"))
        atom_idx = sub_df[sub_df['subject_id'] ==
                          subject_id]['atom_id'].values.astype(int)
        Z_temp.append(cdl_model.z_hat_[:, atom_idx, :])
        min_n_times_valid = min(min_n_times_valid, Z_temp[-1].shape[2])
        D_temp.append(cdl_model.D_hat_[atom_idx, :, :])

    # combine z and d vectors
    Z = Z_temp[0][:, :, :min_n_times_valid]
    D = D_temp[0]
    for this_z, this_d in zip(Z_temp[1:], D_temp[1:]):
        this_z = this_z[:, :, :min_n_times_valid]
        Z = np.concatenate((Z, this_z), axis=1)
        D = np.concatenate((D, this_d), axis=0)

    print(Z.shape)
    print(D.shape)

    X = construct_X_multi(Z, D)

    # select only grad channels
    meg_indices = mne.pick_types(info, meg='grad')
    info = mne.pick_info(info, meg_indices)

    n_times_atom = D.shape[-1]

    return X, info, n_times_atom


def compute_mean_atom(df_topomaps, use_batch_cdl):
    """

    """
    # run CSC with one atom
    cdl_params = {
        'n_atoms': 1,
        'rank1': True, 'uv_constraint': 'separate',
        'window': True,  # in Tim's: False
        'unbiased_z_hat': True,  # in Tim's: False
        'D_init': 'chunk',
        'lmbd_max': 'scaled', 'reg': 0.2,
        'n_iter': 100, 'eps': 1e-5,
        'solver_z': 'lgcd',
        'solver_z_kwargs': {'tol': 1e-3, 'max_iter': 1000},
        'solver_d': 'alternate_adaptive',
        'solver_d_kwargs': {'max_iter': 300},
        'sort_atoms': True,
        'verbose': 1,
        'random_state': 0,
        'use_batch_cdl': use_batch_cdl,
        'n_splits': 1,
        'n_jobs': 5
    }

    def procedure(class_label):
        new_rows = []

        for method in ['cah', 'kmeans']:
            # Reconstruct signal for a given class
            X, info, n_times_atom = reconstruct_class_signal(
                df_topomaps, method, label=class_label)
            cdl_params['n_times_atom'] = n_times_atom
            cdl_model, _ = run_csc(X, **cdl_params)
            # append dataframe
            new_rows.append({'class_label': class_label,
                             'method': method,
                             'u_hat': cdl_model.u_hat_[0],
                             'v_hat': cdl_model.v_hat_[0]})
            # plot "mean" atom
            mne.viz.plot_topomap(data=cdl_model.u_hat_[
                                 0], pos=info, show=False)
            plt.title("Mean atom for age group %i, %s clustering, class %i" %
                      (AGE_GROUP, method, class_label))
            fig_name = 'age_group_' + str(AGE_GROUP) + "_" + \
                method + "_class_" + str(class_label)
            plt.savefig('results_mean_atom/' + fig_name + '.pdf')
            plt.savefig('results_mean_atom/' + fig_name + '.png')
            plt.close()

        return new_rows

    new_rows = Parallel(n_jobs=6, verbose=1)(
        delayed(procedure)(class_label) for class_label in range(n_clusters))

    df_mean = pd.DataFrame()
    for new_row in new_rows:
        df_mean = df_mean.append(new_row, ignore_index=True)

    return df_mean


def temp(df, col_label, cdl_params):
    """
    df : pandas.DataFrame
        the clustering dataframe where each row is an atom, and has at least
        the folowing columns :
            subject_id : the participant id associated with the atom
            u_hat : the topomap vector of the atom
            v_hat : the temporal pattern of the atom
            col_label : the cluster result

    col_label : str
        the name of the column that contains the cultering result

    cdl_params : dict
        the CDL parameters to use to compute the mean atom

    """

    def procedure(class_label):
        # Reconstruct signal for a given class
        X, info, n_times_atom = reconstruct_class_signal(
            df_topomaps, method, label=class_label)
        cdl_params['n_times_atom'] = n_times_atom
        cdl_model, _ = run_csc(X, **cdl_params)
        # append dataframe
        new_row = {'class_label': class_label,
                   'method': method,
                   'u_hat': cdl_model.u_hat_[0],
                   'v_hat': cdl_model.v_hat_[0]}
        # plot "mean" atom
        mne.viz.plot_topomap(data=cdl_model.u_hat_[
            0], pos=info, show=False)
        plt.title("Mean atom for age group %i, %s clustering, class %i" %
                  (AGE_GROUP, method, class_label))
        fig_name = 'age_group_' + str(AGE_GROUP) + "_" + \
            method + "_class_" + str(class_label)
        plt.savefig('results_mean_atom/' + fig_name + '.pdf')
        plt.savefig('results_mean_atom/' + fig_name + '.png')
        plt.close()

        return new_row


def plot_global_class(df, df_mean=None, method='cah', class_label=0):
    """

    """
    sub_df = df[df['labels_' + method] == class_label]
    subjects = set(sub_df['subject_id'].values)
    n_atoms = sub_df.shape[0]

    if df_mean is not None:
        u_mean = df_mean[(df_mean['method'] == method) & (
            df_mean['class_label'] == class_label)]['u_hat'].values[0]
        plot_mean = True
    else:
        plot_mean = False

    # plot topomaps
    n_columns = min(6, n_atoms)
    split = int(np.ceil(n_atoms / n_columns))
    if (n_atoms % n_columns) in [0, (n_columns - 1)]:
        split += 1
    figsize = (4 * n_columns, 3 * split)
    fig, axes = plt.subplots(split, n_columns, figsize=figsize)
    axes = np.atleast_2d(axes)

    for ii in range(n_atoms):
        row = sub_df.iloc[ii]
        subject_id, atom_id = row['subject_id'], int(row['atom_id'])

        # get CDL results for the subject_id
        file_name = RESULTS_DIR / subject_id / 'CSCraw_0.5s_20atoms.pkl'
        cdl_model, info, _, _ = pickle.load(open(file_name, "rb"))
        u_hat_ = cdl_model.u_hat_
        # select only grad channels
        meg_indices = mne.pick_types(info, meg='grad')
        info = mne.pick_info(info, meg_indices)

        i_row, i_col = ii // n_columns, ii % n_columns
        it_axes = iter(axes[i_row:(i_row + 1), i_col])

        ax = next(it_axes)
        ax.set_title('Atom % d Subject %s' % (atom_id, subject_id), pad=0)

        mne.viz.plot_topomap(
            data=u_hat_[atom_id], pos=info, axes=ax, show=False)

    if plot_mean:
        # plot mean atom
        ax = axes[-1, -1]
        ax.set_title('Mean atom', pad=0)
        mne.viz.plot_topomap(
            data=u_mean, pos=info, axes=ax, show=False)

    # save figure
    fig.tight_layout()
    fig_name = 'global_figure_age_group_' + str(AGE_GROUP) + "_" + \
        method + "_class_" + str(class_label)
    plt.savefig('results_mean_atom/' + fig_name + '.pdf')
    plt.show()


# For each subject, we create the atom clustering based on topomaps correlation
df_topomaps, data_columns = get_df_topomaps(AGE_GROUP)

# Select number of classes to keep
plt.title("CAH on CAM-Can topomaps for age group" + str(AGE_GROUP))
dendrogram(linkage(np.array(df_topomaps[data_columns]), method='ward'),
           orientation='right', truncate_mode='level', p=5)
plt.show()

# %%
n_clusters = 6
# compute clustering on topomaps
df_topomaps = df_culstering(df_topomaps, data_columns, n_clusters)
df_mean = compute_mean_atom(df_topomaps, use_batch_cdl=False)

for method in ['cah', 'kmeans']:
    for class_label in range(n_clusters):
        plot_global_class(df_topomaps, df_mean, method, class_label)

# %%
