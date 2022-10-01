# %%
from utils_csc import compute_distance_matrix
from utils_plot import plot_atoms_single_sub
import numpy as np
import pandas as pd
import re
from config import N_JOBS
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Memory, Parallel, delayed
import pickle

import mne
print(mne.__version__)


SAVE = True


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


def compute_n_groups(y_pred):
    """

    """
    # all outliers points are labelled with "-1"
    n_noise_point = len(y_pred[y_pred == -1])
    # make sure we count outliers separetly and not twice
    n_groups = len(np.unique(y_pred)) + n_noise_point - (n_noise_point > 0)
    return n_groups


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
               'n_groups': compute_n_groups(y_pred)}

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
    title = 'results_dbscan/heatmap'
    if exclude_subs is not None:
        title += '_with_exclusion'
    title += '.jpg'
    if SAVE:
        plt.savefig(title)
    plt.show()

    return df_dbscan


def run_dbscan_group_analysis(atom_df, eps=0.1, min_samples=1, plot=False):
    """Compute DBscan on each subjects in atom_df.

    """

    def procedure(subject_id):

        D = compute_distance_matrix(
            atom_df[atom_df['subject_id'] == subject_id])

        y_pred = DBSCAN(eps=eps, min_samples=min_samples,
                        metric='precomputed').fit_predict(D)
        row = {'subject_id': subject_id, 'eps': eps,
               'min_samples': min_samples,
               'n_groups': compute_n_groups(y_pred)}

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
        plt.title(
            f'Number of groups obtain on single subject, eps = {eps} and min_samples = {min_samples}')
        plt.savefig(
            f'results_dbscan/group_analysis_eps_{eps}_min_samples_{min_samples}.jpg')
        plt.show()

    return df_dbscan, df_group_analysis


# %%
# exclusion list in Lindsey's
exclude_subs_dal = ['CC420061', 'CC121397', 'CC420396', 'CC420348', 'CC320850',
                    'CC410325', 'CC121428', 'CC110182', 'CC420167', 'CC420261',
                    'CC322186', 'CC220610', 'CC221209', 'CC220506', 'CC110037',
                    'CC510043', 'CC621642', 'CC521040', 'CC610052', 'CC520517',
                    'CC610469', 'CC720497', 'CC610292', 'CC620129', 'CC620490']

# atom_df = pickle.load(open('all_atoms_info.pkl', "rb"))
atom_df = pd.read_csv('atomData_v2.csv')
pattern = re.compile("[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
if type(atom_df['u_hat'][0]) == str:
    atom_df['u_hat'] = atom_df['u_hat'].apply(
        lambda x: np.array(pattern.findall(x)).astype(float))
    atom_df['v_hat'] = atom_df['v_hat'].apply(
        lambda x: np.array(pattern.findall(x)).astype(float))
print(atom_df)

# %%
# get_eps_rot(atom_df, p=90)
# _ = run_dbscan_heatmap(atom_df, exclude_subs=None)

# %% For a list of hyper-parameters, compute the DBscan for each subject
# separatly and plot the repartition of subjects in function of the number of
# unique groups they obtained with DBscan

list_eps = np.linspace(0.05, 0.8, 16)
list_min_samples = [1]

df_hp_analysis = pd.DataFrame()


for ii, min_sample in enumerate(list_min_samples):
    for jj, eps in enumerate(list_eps):
        df_dbscan, df_group_analysis = run_dbscan_group_analysis(
            atom_df, eps=eps, min_samples=min_sample)
        df_hp_analysis = df_hp_analysis.append(df_dbscan, ignore_index=True)

if SAVE:
    df_hp_analysis.to_csv('results_dbscan/df_hp_analysis.csv', index=False)
# %%

df_hp_analysis = pd.read_csv('results_dbscan/df_hp_analysis.csv')

list_eps = np.linspace(0.05, 0.45, 9)
min_samples = 1

n_col = 3
fig, axes = plt.subplots(3, n_col)
axes = np.atleast_2d(axes)

for ii, eps in enumerate(list_eps):
    df_dbscan = df_hp_analysis[(df_hp_analysis['eps'] == eps) & (
        df_hp_analysis['min_samples'] == min_samples)]
    df_group_analysis = df_dbscan.groupby('n_groups')\
        .agg({'subject_id': 'count'})\
        .rename(columns={'subject_id': 'nunique_subject_id'})\
        .reset_index()
    ax = axes[ii//n_col, ii % n_col]
    ax.hist(df_group_analysis.n_groups,
            weights=df_group_analysis.nunique_subject_id,
            bins=np.arange(1, 22) - 0.5, range=(1, 21),
            label="nunique_subject_id")
    ax.set_title(f"$\epsilon$ = {eps:.2}")
    # ax.set_xticks(range(1, 21))
    ax.set_xlim([0.5, 20.5])

fig.tight_layout()
if SAVE:
    plt.savefig('results_dbscan/hp_group_analysis.jpg')
    plt.savefig('results_dbscan/hp_group_analysis.pdf')
plt.show()


# plot single subject analysis

# %%
eps = 0.1
min_samples = 1
min_n_groups = 14
df_dbscan = df_hp_analysis[(df_hp_analysis['eps'] == eps) & (
    df_hp_analysis['min_samples'] == min_samples)]
df_group_analysis = df_dbscan.groupby('n_groups')\
    .agg({'subject_id': 'count'})\
    .rename(columns={'subject_id': 'nunique_subject_id'})\
    .reset_index()

plt.hist(df_group_analysis.n_groups,
         weights=df_group_analysis.nunique_subject_id,
         bins=np.arange(1, 22) - 0.5, range=(1, 21),
         label="nunique_subject_id")
plt.vlines(x=min_n_groups, ymin=0, ymax=df_group_analysis['nunique_subject_id'].max(),
           linestyles='--')
plt.title(f"$\epsilon$ = {eps:.2}")
plt.xticks(range(1, 21))
plt.xlim([0.5, 20.5])
plt.savefig('results_dbscan/single_sub_exclude_hist.pdf')
plt.savefig('results_dbscan/single_sub_exclude_hist.png')
plt.show()

exclude_subs = df_dbscan[df_dbscan['n_groups']
                         < min_n_groups]['subject_id'].values
print(
    f"Excluding subjects with strictly less than {min_n_groups} clusters results in the exclusion of {len(exclude_subs)} subjects: \n{exclude_subs}")

both_exclude = [
    subject_id for subject_id in exclude_subs if subject_id in exclude_subs_dal]
print(
    f'including {len(both_exclude)} that were previoulsy excluded: \n{both_exclude}')

excluded_subjects = atom_df[atom_df['subject_id'].isin(
    exclude_subs)].reset_index()
excluded_subjects = excluded_subjects[['subject_id', 'age', 'sex']]\
    .drop_duplicates()\
    .sort_values('subject_id')\
    .reset_index(drop=True)
print(excluded_subjects.to_latex())

# %%
for subject_id in df_dbscan[df_dbscan['n_groups']
                            == 13]['subject_id'].values:
    try:
        plot_atoms_single_sub(
            atom_df, subject_id, sfreq=150., plot_psd=False, plot_dipole=False, save_dir='results_dbscan')
        print(f'Plot atoms for subject {subject_id}')
    except:
        pass


# %%

_ = run_dbscan_heatmap(atom_df, exclude_subs=exclude_subs)

# %%
eps = 0.1
min_samples = 2
subject_id = 'CC723395'
D = compute_distance_matrix(
    atom_df[atom_df['subject_id'] == subject_id])

y_pred = DBSCAN(eps=eps, min_samples=min_samples,
                metric='precomputed').fit_predict(D)
print(y_pred)
# %%
# Final Clustering

eps = 0.1
min_samples = 2
atom_df_exclude = atom_df[~atom_df['subject_id'].isin(
    exclude_subs)].reset_index()
D = compute_distance_matrix(atom_df_exclude)

y_pred = DBSCAN(eps=eps, min_samples=min_samples,
                metric='precomputed').fit_predict(D)
print(y_pred)
print(compute_n_groups(y_pred))

atom_df_exclude['final_label'] = y_pred
clustering_analysis = atom_df_exclude.groupby('final_label')\
    .agg({'subject_id': 'count'})\
    .rename(columns={'subject_id': 'nunique_subject_id'})\
    .reset_index()

# %%
