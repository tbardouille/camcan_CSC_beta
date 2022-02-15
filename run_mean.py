import pandas as pd
import pickle

import mne

from config import RESULTS_DIR
from utils_csc import double_correlation_clustering, get_df_mean, plot_mean_atom


# all_atoms_info = pd.read_csv('./all_atoms_info.csv')
atom_df = pickle.load(open('all_atoms_info.pkl', "rb"))

exclude_subs = ['CC420061', 'CC121397', 'CC420396', 'CC420348', 'CC320850',
                'CC410325', 'CC121428', 'CC110182', 'CC420167', 'CC420261',
                'CC322186', 'CC220610', 'CC221209', 'CC220506', 'CC110037',
                'CC510043', 'CC621642', 'CC521040', 'CC610052', 'CC520517',
                'CC610469', 'CC720497', 'CC610292', 'CC620129', 'CC620490']

atom_groups, group_summary = double_correlation_clustering(
    atom_df, u_thresh=0.4, v_thresh=0.4, exclude_subs=exclude_subs,
    output_dir=None)

print(group_summary)

# select big enough groups
total_subjects = group_summary['count_atom_id'].sum() / 20
threshold_group = 1/3
group_id = group_summary[group_summary['nunique_subject_id']
                         > threshold_group * total_subjects]['group_number'].values
# compute mean atom only for bigger groups
df = pd.merge(atom_df, atom_groups, how="right", on=["subject_id", "atom_id"])
df_mean = get_df_mean(df=df[df['group_number'].isin(group_id)],
                      col_label='group_number')
# get info
subject_id = atom_groups['subject_id'].values[0]
file_name = RESULTS_DIR / subject_id / 'CSCraw_0.5s_20atoms.pkl'
_, info, _, _ = pickle.load(open(file_name, "rb"))
meg_indices = mne.pick_types(info, meg='grad')
info = mne.pick_info(info, meg_indices)
# plot mean atoms
fig = plot_mean_atom(df_mean, info, plot_psd=True)
