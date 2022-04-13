import pandas as pd
import numpy as np
import re
from pathlib import Path
import pickle

from sklearn.cluster import DBSCAN

from config import N_JOBS
from utils_csc import compute_distance_matrix

# exclusion list
exclude_subs = ['CC510434', 'CC521040', 'CC420061', 'CC610469', 'CC121397',
                'CC720497', 'CC420396', 'CC420348', 'CC721052', 'CC610052',
                'CC320850', 'CC410325', 'CC121428', 'CC520560', 'CC520517',
                'CC110182', 'CC420167', 'CC620129', 'CC620490', 'CC420261',
                'CC220610', 'CC221209', 'CC620005', 'CC121144', 'CC221040',
                'CC220519', 'CC320893', 'CC220506', 'CC510043']

# atom_df = pickle.load(open('all_atoms_info.pkl', "rb"))
atom_df = pd.read_csv('atomData_v2.csv')
pattern = re.compile("[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
if type(atom_df['u_hat'][0]) == str:
    atom_df['u_hat'] = atom_df['u_hat'].apply(
        lambda x: np.array(pattern.findall(x)).astype(float))
    atom_df['v_hat'] = atom_df['v_hat'].apply(
        lambda x: np.array(pattern.findall(x)).astype(float))
print(atom_df)

# exclude subjects
atom_df = atom_df[~atom_df['subject_id'].isin(
    exclude_subs)].reset_index()
# compute distance matrix and save it
if not Path('./dbscan_distance_matrix.pkl').exists():
    D = compute_distance_matrix(atom_df)
    pickle.dump(D, open('./dbscan_distance_matrix.pkl', "wb"))
else:
    D = pickle.load(open('./dbscan_distance_matrix.pkl', "rb"))

# apply DBScan
eps = 0.1
min_samples = 1
y_pred = DBSCAN(eps=eps, min_samples=min_samples,
                metric='precomputed').fit_predict(D)
atom_df['label'] = y_pred
