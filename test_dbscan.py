# %%


from utils_plot import plot_atoms_single_sub
from utils_csc import compute_distance_matrix
from config import N_JOBS
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Memory, Parallel, delayed
import pickle

import mne
print(mne.__version__)


# dip = get_subject_dipole(subject_id='CC110606')
# print(dip.pos)


def get_eps_rot(atom_df, p=90):
    """Compute the rule of thumb (rot) for DBscan hyper-parameter eps.
    We want esp such that (p/100)% of the subjects have their nearest neightbour 
    within a radius of  esp.

    """
    D = compute_distance_matrix(atom_df)

    distances = np.sort(D, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    q = int(atom_df.shape[0] * p / 100)
    plt.vlines(q, 0, 1, linestyles='--', label=f'{p}%')
    plt.legend()
    plt.show()

    eps = round(distances[q], 2)
    print(f"epsilon choice so that 90% of individuals have their nearest \
            neightbour in less than epsilon: {eps}")

    return eps


def run_dbscan_heatmap(atom_df, exclude_subs=None):
    """Compute DBscan on atom_df (eventually after removing subjects in
    exclude_subs) for different hyper-parameters, and plot the heatmap of the
    number of distinct groups obtains, in function of the 2 hyper-parameters

    """
    if exclude_subs is not None:
        atom_df = atom_df[~atom_df['subject_id'].isin(
            exclude_subs)].reset_index()
    D = compute_distance_matrix(atom_df)

    list_esp = np.linspace(0.1, 0.5, 9)
    list_min_samples = [1, 2, 3]

    def procedure(eps, min_samples):

        y_pred = DBSCAN(eps=eps, min_samples=min_samples,
                        metric='precomputed').fit_predict(D)
        row = {'eps': eps, 'min_samples': min_samples,
               'n_groups': len(np.unique(y_pred))}

        return row

    new_rows = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(procedure)(eps, min_samples)
        for eps in list_esp for min_samples in list_min_samples)

    df_dbscan = pd.DataFrame()
    for new_row in new_rows:
        df_dbscan = df_dbscan.append(new_row, ignore_index=True)

    n_groups = df_dbscan.pivot("eps", "min_samples", "n_groups")
    ax = sns.heatmap(n_groups, annot=True)
    title = 'Number of groups obtains with DBScan'
    if exclude_subs is not None:
        title += ' (with exclusion)'

    ax.set_title(title)
    ax.set_ylabel(r'$\varepsilon$')
    ax.set_xlabel(r"min samples")
    plt.show()

    return df_dbscan


def run_dbscan_group_analysis(atom_df, eps=0.2, min_samples=2, plot=False):
    """Compute DBscan on each subjects in atom_df.

    """

    def procedure(subject_id):

        D = compute_distance_matrix(
            atom_df[atom_df['subject_id'] == subject_id])

        y_pred = DBSCAN(eps=eps, min_samples=min_samples,
                        metric='precomputed').fit_predict(D)
        row = {'subject_id': subject_id, 'eps': eps,
               'min_samples': min_samples,
               'n_groups': len(np.unique(y_pred))}

        return row

    new_rows = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(procedure)(this_subject_id)
        for this_subject_id in atom_df['subject_id'].unique())

    df_dbscan = pd.DataFrame()
    for new_row in new_rows:
        df_dbscan = df_dbscan.append(new_row, ignore_index=True)

    df_group_analysis = df_dbscan.groupby('n_groups')\
        .agg({'subject_id': 'count'})\
        .rename(columns={'subject_id': 'nunique_subject_id'})\
        .reset_index()

    if plot:
        df_group_analysis.plot(x='n_groups', y='nunique_subject_id')
        plt.title(f'Number of groups obtain on single subject, \
            with eps = {eps} and min_samples = {min_samples}')
        plt.show()

    return df_dbscan, df_group_analysis


# exclusion list in Lindsey's
exclude_subs_dal = ['CC420061', 'CC121397', 'CC420396', 'CC420348', 'CC320850',
                    'CC410325', 'CC121428', 'CC110182', 'CC420167', 'CC420261',
                    'CC322186', 'CC220610', 'CC221209', 'CC220506', 'CC110037',
                    'CC510043', 'CC621642', 'CC521040', 'CC610052', 'CC520517',
                    'CC610469', 'CC720497', 'CC610292', 'CC620129', 'CC620490']

atom_df = pickle.load(open('all_atoms_info.pkl', "rb"))

# %%
get_eps_rot(atom_df, p=90)
_ = run_dbscan_heatmap(atom_df, exclude_subs=None)

# %% For a list of hyper-parameters, compute the DBscan for each subject
# separatly and plot the repartition of subjects in function of the number of
# unique groups they obtained with DBscan

list_eps = [0.1, 0.2, 0.3, 0.4]
list_min_samples = [1, 2, 3]

fig, axes = plt.subplots(len(list_min_samples), len(list_eps))
axes = np.atleast_2d(axes)

for ii, min_sample in enumerate(list_min_samples):
    for jj, eps in enumerate(list_eps):
        df_dbscan, df_group_analysis = run_dbscan_group_analysis(
            atom_df, eps=eps, min_samples=min_sample)
        ax = axes[ii, jj]
        if jj == 0:
            ax.set_ylabel(f'min_samples = {min_sample}')
        if ii == 0:
            ax.set_title(f'eps = {eps}')
        df_group_analysis.plot(
            x='n_groups', y='nunique_subject_id', ax=ax, legend=False)

fig.tight_layout()
plt.show()
# %%
eps = 0.1
min_samples = 1
df_dbscan, df_group_analysis = run_dbscan_group_analysis(
    atom_df, eps=eps, min_samples=min_samples, plot=True)
# %%
min_n_groups = 14
exclude_subs = df_dbscan[df_dbscan['n_groups']
                         < min_n_groups]['subject_id'].values
print(f'exclusion list: {exclude_subs}')
subject_id = exclude_subs[0]
print(f'Plot atoms for subject {subject_id}')
fig = plot_atoms_single_sub(
    atom_df, subject_id, sfreq=150., plot_psd=False, plot_dipole=False)
# %%

_ = run_dbscan_heatmap(atom_df, exclude_subs=exclude_subs)

# %%
