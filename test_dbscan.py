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


all_atoms_info = pickle.load(open('all_atoms_info.pkl', "rb"))

# %%
atom_df = all_atoms_info.copy()
D = compute_distance_matrix(atom_df)
print(D.min(), D.max())

distances = np.sort(D, axis=0)
distances = distances[:, 1]
plt.plot(distances)
p = 90
q = int(all_atoms_info.shape[0] * p / 100)
plt.vlines(q, 0, 1, linestyles='--', label=f'{p}%')
plt.legend()
plt.show()

eps = round(distances[q], 2)
print(
    f"epsilon choice so that 90% of individuals have their nearest neightbour in less than epsilon: {eps}")

# %%
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

# %%
n_groups = df_dbscan.pivot("eps", "min_samples", "n_groups")
ax = sns.heatmap(n_groups, annot=True)
ax.set_title('Number of groups obtains with DBScan')
ax.set_ylabel(r'$\varepsilon$')
ax.set_xlabel(r"min samples")
plt.show()

# %%
