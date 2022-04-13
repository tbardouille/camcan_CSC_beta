# %%
import pandas as pd
import pickle

import mne

from config import RESULTS_DIR, CDL_PARAMS, get_cdl_pickle_name
from utils_csc import double_correlation_clustering, get_df_mean, reconstruct_class_signal, run_csc
from utils_plot import plot_mean_atom
# %%

# all_atoms_info = pd.read_csv('./all_atoms_info.csv')
atom_df = pickle.load(open('all_atoms_info.pkl', "rb"))

exclude_subs = ['CC420061', 'CC121397', 'CC420396', 'CC420348', 'CC320850',
                'CC410325', 'CC121428', 'CC110182', 'CC420167', 'CC420261',
                'CC322186', 'CC220610', 'CC221209', 'CC220506', 'CC110037',
                'CC510043', 'CC621642', 'CC521040', 'CC610052', 'CC520517',
                'CC610469', 'CC720497', 'CC610292', 'CC620129', 'CC620490']

print("Start computing clustering")
atom_groups, group_summary = double_correlation_clustering(
    atom_df, u_thresh=0.4, v_thresh=0.4, exclude_subs=exclude_subs,
    output_dir=RESULTS_DIR)
print(group_summary)

# %% select big enough groups
total_subjects = group_summary['count_atom_id'].sum() / CDL_PARAMS['n_atoms']
threshold_group = 1/3
group_id = group_summary[group_summary['nunique_subject_id']
                         > threshold_group * total_subjects]['group_number'].values
print(group_id)
df = pd.merge(atom_df, atom_groups, how="right", on=["subject_id", "atom_id"])
# %% compute mean atom only for bigger groups
df_mean = get_df_mean(df=df[df['group_number'].isin(group_id)],
                      col_label='group_number')
# get info
subject_id = atom_groups['subject_id'].values[0]
file_name = RESULTS_DIR / subject_id / 'CSCraw_0.5s_20atoms.pkl'
_, info, _, _ = pickle.load(open(file_name, "rb"))
print("Start computing mean atom")
info = mne.pick_info(info, mne.pick_types(info, meg='grad'))
# plot mean atoms
fig = plot_mean_atom(df_mean, info, plot_psd=True)

# %%

cdl_params = CDL_PARAMS.copy()
cdl_params.update(n_atoms=1, n_splits=1)
col_label = 'group_number'

def procedure(label):
    # Reconstruct signal for a given class
    X, n_times_atom = reconstruct_class_signal(
        df=df[df[col_label] == label], results_dir=RESULTS_DIR)
    cdl_params['n_times_atom'] = n_times_atom
    cdl_model, z_hat = run_csc(X, **cdl_params)
    # append dataframe
    new_row = {col_label: label,
                'u_hat': cdl_model.u_hat_[0],
                'v_hat': cdl_model.v_hat_[0],
                'z_hat': z_hat,
                'n_times_atom': n_times_atom}

    return new_row

new_row = procedure(label=6)

subject_id = df['subject_id'][0]  # 'CC110606'
file_name = RESULTS_DIR / subject_id / get_cdl_pickle_name()
_, info, _, _ = pickle.load(open(file_name, "rb"))
evoked = mne.EvokedArray(new_row['u_hat'], info)

#Compute noise covariance
empty_room = mne.read_epochs(emptyroomFif)
noise_cov = mne.compute_covariance(empty_room, tmin=0, tmax=None)
     
#Fit dipoles
dip = mne.fit_dipole(evoked, noise_cov, bemFif, transFif,  verbose=False)[0]
dip.plot_locations(transFif,'fsaverage',subjectsDir)
# %%
