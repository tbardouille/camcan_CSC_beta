# %%
from alphacsc.utils.convolution import construct_X, construct_X_multi
from alphacsc.utils.dictionary import get_D
from alphacsc import BatchCDL, GreedyCDL
import alphacsc
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import os

from sklearn import cluster
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt

from joblib import Memory, Parallel, delayed

import mne
from utils_csc import get_atom_df, correlation_clustering_atoms, get_df_mean
from utils_plot import plot_mean_atom
from config import CDL_PARAMS, RESULTS_DIR, PARTICIPANTS_FILE

from utils_csc import get_atom_df, correlation_clustering_atoms, get_df_mean
from utils_plot import plot_mean_atom
from config import CDL_PARAMS, RESULTS_DIR, PARTICIPANTS_FILE  # , get_paths


# %%
# read clustering results
OUTPUT_DIR = '/media/NAS/lpower/CSC/results/u_'
threshold = u_thresh = v_thresh = 0.4

csv_dir = OUTPUT_DIR + \
    str(u_thresh) + '_v_' + str(v_thresh) + '_groupSummary.csv'
if not os.path.exists(csv_dir):
    print(f'{csv_dir} does not exist')
    # atom_df = get_atom_df(RESULTS_DIR, PARTICIPANTS_FILE)
    # groupSummary, atomGroups = correlation_clustering_atoms(
    #     atom_df, threshold=threshold, output_dir=OUTPUT_DIR)

# %%
groupSummary = pd.read_csv(csv_dir)

# select big enough groups
threshold_group = .25
total_subjects = groupSummary['Number of Atoms'].sum() / 20
group_id = groupSummary[groupSummary['Number of Subjects']
                        > threshold_group * total_subjects]['Group Number'].values
print(total_subjects)
# Save atomGroups to dataframe
csv_dir = OUTPUT_DIR + \
    str(u_thresh) + '_v_' + str(v_thresh) + '_atomGroups.csv'
atomGroups = pd.read_csv(csv_dir)

atomGroups.rename(columns={'Subject ID': 'subject_id'}, inplace=True)
atomGroups.rename(columns={'Atom number': 'atom_id'}, inplace=True)
clustering_df = atomGroups[atomGroups['Group number'].isin(group_id)]
df_mean = get_df_mean(clustering_df, col_label='Group number',
                      cdl_params=CDL_PARAMS, results_dir=RESULTS_DIR, n_jobs=6)

# get info
subject_id = atomGroups['subject_id'].values[0]
file_name = RESULTS_DIR / subject_id / 'CSCraw_0.5s_20atoms.pkl'
_, info, _, _ = pickle.load(open(file_name, "rb"))
meg_indices = mne.pick_types(info, meg='grad')
info = mne.pick_info(info, meg_indices)


fig = plot_mean_atom(df_mean, info, plot_psd=True)

# Paths for Cam-CAN dataset
# DATA_DIR = Path("/storage/store/data/")
# BEM_DIR = DATA_DIR / "camcan-mne/freesurfer"
# TRANS_DIR = DATA_DIR / "camcan-mne/trans"
# BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"  # Root path to raw BIDS files
# PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"

# path to all participant CSC results (local)
# RESULTS_DIR = Path('./results_csc')
# SUBJECT_DIRS = [f for f in RESULTS_DIR.iterdir() if not f.is_file()]
# PARTICIPANTS_FILE = Path('./participants.tsv')

# # AGE_GROUP = 1


# atom_df = get_atom_df(RESULTS_DIR, PARTICIPANTS_FILE)

# # data_columns = ['u_hat']
# n_captors = len(atom_df['u_hat'].iloc[0])
# data_columns = [f'u_hat_{i}' for i in range(n_captors)]
# split_df = pd.DataFrame(atom_df['u_hat'].tolist(),
#                         columns=data_columns)
# atom_df = pd.concat([atom_df, split_df], axis=1)

# # get info
# subject_id = atom_df['subject_id'].values[0]
# file_name = RESULTS_DIR / subject_id / 'CSCraw_0.5s_20atoms.pkl'
# _, info, _, _ = pickle.load(open(file_name, "rb"))
# meg_indices = mne.pick_types(info, meg='grad')
# info = mne.pick_info(info, meg_indices)

# # %% Select number of classes to keep
# plt.title("CAH on CAM-Can topomaps vectors")
# dendrogram(linkage(np.array(atom_df[data_columns]), method='ward'),
#            orientation='right', truncate_mode='level', p=5)
# plt.show()

# # %% Apply clustering
# n_clusters = 6
# culstering_df = culstering_cah_kmeans(atom_df, data_columns)

# # %% Recompute signal and compute mean atom
# use_batch_cdl = False
# cdl_params = {
#     'rank1': True, 'uv_constraint': 'separate',
#     'window': True,  # in Tim's: False
#     'unbiased_z_hat': True,  # in Tim's: False
#     'D_init': 'chunk',
#     'lmbd_max': 'scaled', 'reg': 0.2,
#     'n_iter': 100, 'eps': 1e-5,
#     'solver_z': 'lgcd',
#     'solver_z_kwargs': {'tol': 1e-3, 'max_iter': 1000},
#     'solver_d': 'alternate_adaptive',
#     'solver_d_kwargs': {'max_iter': 300},
#     'sort_atoms': True,
#     'verbose': 1,
#     'random_state': 0,
#     'use_batch_cdl': use_batch_cdl,
#     'n_splits': 1,
#     'n_jobs': 5
# }
# df_mean = get_df_mean(culstering_df, col_label='labels_cah',
#                       cdl_params=cdl_params, results_dir=RESULTS_DIR, n_jobs=6)
# plot_mean_atom(df_mean, info)
# # %%


# def get_df_topomaps(age_group):
#     """

#     """
#     # get the subject's folders that correspond to the age group
#     age_group_dir = [subject_dir for subject_dir in SUBJECT_DIRS
#                      if subject_dir.name[:3] == ('CC' + str(age_group))]

#     # For every subject in the age group, get the associated topomap vector
#     age_group_u = {this_dir.name: pickle.load(
#         open(this_dir / 'CSCraw_0.5s_20atoms.pkl', "rb"))[0].u_hat_
#         for this_dir in age_group_dir}

#     # Create a dataframe containing all topomaps
#     df_topomaps = pd.DataFrame()
#     for subject_id, u_hat in age_group_u.items():
#         for i, u in enumerate(u_hat):
#             new_row = {i: u[i] for i in range(len(u))}
#             new_row['subject_id'] = subject_id
#             new_row['atom_id'] = int(i)
#             df_topomaps = df_topomaps.append(new_row, ignore_index=True)

#     data_columns = [i for i in range(len(u))]

#     return df_topomaps, data_columns


# def plot_topomaps(subject_id, atom_idx):
#     # get CDL results for the subject_id
#     file_name = RESULTS_DIR / subject_id / 'CSCraw_0.5s_20atoms.pkl'
#     cdl_model, info, _, _ = pickle.load(open(file_name, "rb"))
#     u_hat_ = cdl_model.u_hat_
#     # select only grad channels
#     meg_indices = mne.pick_types(info, meg='grad')
#     info = mne.pick_info(info, meg_indices)
#     # plot topomaps
#     n_columns = min(4, len(atom_idx))
#     split = int(np.ceil(len(atom_idx) / n_columns))
#     figsize = (4 * n_columns, 3 * split)
#     fig, axes = plt.subplots(split, n_columns, figsize=figsize)
#     axes = np.atleast_2d(axes)

#     for ii, kk in enumerate(atom_idx):
#         i_row, i_col = ii // n_columns, ii % n_columns
#         it_axes = iter(axes[i_row:(i_row + 1), i_col])

#         ax = next(it_axes)
#         ax.set_title('Atom % d Subject %s' % (kk, subject_id), pad=0)

#         mne.viz.plot_topomap(data=u_hat_[kk], pos=info, axes=ax, show=False)

#     fig.tight_layout()
#     plt.show()


# def plot_global_class(df, df_mean=None, method='cah', class_label=0):
#     """

#     """
#     sub_df = df[df['labels_' + method] == class_label]
#     subjects = set(sub_df['subject_id'].values)
#     n_atoms = sub_df.shape[0]

#     if df_mean is not None:
#         u_mean = df_mean[(df_mean['method'] == method) & (
#             df_mean['class_label'] == class_label)]['u_hat'].values[0]
#         plot_mean = True
#     else:
#         plot_mean = False

#     # plot topomaps
#     n_columns = min(6, n_atoms)
#     split = int(np.ceil(n_atoms / n_columns))
#     if (n_atoms % n_columns) in [0, (n_columns - 1)]:
#         split += 1
#     figsize = (4 * n_columns, 3 * split)
#     fig, axes = plt.subplots(split, n_columns, figsize=figsize)
#     axes = np.atleast_2d(axes)

#     for ii in range(n_atoms):
#         row = sub_df.iloc[ii]
#         subject_id, atom_id = row['subject_id'], int(row['atom_id'])

#         # get CDL results for the subject_id
#         file_name = RESULTS_DIR / subject_id / 'CSCraw_0.5s_20atoms.pkl'
#         cdl_model, info, _, _ = pickle.load(open(file_name, "rb"))
#         u_hat_ = cdl_model.u_hat_
#         # select only grad channels
#         meg_indices = mne.pick_types(info, meg='grad')
#         info = mne.pick_info(info, meg_indices)

#         i_row, i_col = ii // n_columns, ii % n_columns
#         it_axes = iter(axes[i_row:(i_row + 1), i_col])

#         ax = next(it_axes)
#         ax.set_title('Atom % d Subject %s' % (atom_id, subject_id), pad=0)

#         mne.viz.plot_topomap(
#             data=u_hat_[atom_id], pos=info, axes=ax, show=False)

#     if plot_mean:
#         # plot mean atom
#         ax = axes[-1, -1]
#         ax.set_title('Mean atom', pad=0)
#         mne.viz.plot_topomap(
#             data=u_mean, pos=info, axes=ax, show=False)

#     # save figure
#     fig.tight_layout()
#     fig_name = 'global_figure_age_group_' + str(AGE_GROUP) + "_" + \
#         method + "_class_" + str(class_label)
#     plt.savefig('results_mean_atom/' + fig_name + '.pdf')
#     plt.show()


# # For each subject, we create the atom clustering based on topomaps correlation
# df_topomaps, data_columns = get_df_topomaps(AGE_GROUP)

# # Select number of classes to keep
# plt.title("CAH on CAM-Can topomaps for age group" + str(AGE_GROUP))
# dendrogram(linkage(np.array(df_topomaps[data_columns]), method='ward'),
#            orientation='right', truncate_mode='level', p=5)
# plt.show()

# # %%
# n_clusters = 6
# # compute clustering on topomaps
# df_topomaps = df_culstering(df_topomaps, data_columns, n_clusters)
# df_mean = compute_mean_atom(df_topomaps, use_batch_cdl=False)

# for method in ['cah', 'kmeans']:
#     for class_label in range(n_clusters):
#         plot_global_class(df_topomaps, df_mean, method, class_label)

# %%
