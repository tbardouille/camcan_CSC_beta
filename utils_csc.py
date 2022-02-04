"""
Utils scripts for utils functions 
"""
from codecs import ignore_errors
import scipy.signal as ss
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from joblib import Memory, Parallel, delayed

import mne
from alphacsc import BatchCDL, GreedyCDL
from alphacsc.utils.signal import split_signal
from alphacsc.utils.convolution import construct_X_multi

from config import CDL_PARAMS  # , get_paths

# Paths for Cam-CAN dataset
DATA_DIR = Path("/storage/store/data/")
BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"  # Root path to raw BIDS files
PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"

fifFile = '/transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif'
EPOCH_DIR = '/home/timb/camcan/proc_data/TaskSensorAnalysis_transdef/'

DATA_DIR = Path("/storage/store/data/")
BEM_DIR = DATA_DIR / "camcan-mne/freesurfer"
TRANS_DIR = DATA_DIR / "camcan-mne/trans"

RESULT_DIR = ''  # XXX


def get_paths(subject_id, dal=True):
    if dal:
        # Files for dipole fitting
        subjectsDir = '/home/timb/camcan/subjects'
        transFif = subjectsDir + '/coreg/sub-' + subject_id + '-trans.fif'
        bemFif = subjectsDir + '/sub-' + subject_id + '/bem/sub-' + \
            subject_id + '-5120-bem-sol.fif'
        fifFile = '/transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif'
        epochFif = '/home/timb/camcan/proc_data/TaskSensorAnalysis_transdef/' + \
            subject_id + fifFile
    else:
        # XXX find the good paths for Parital drago server
        bemFif = BEM_DIR / subject_id / \
            'bem' / (subject_id + '-meg-bem.fif')
        subjectsDir + '/sub-' + subject_id + \
            '/bem/sub-' + subject_id + '-5120-bem-sol.fif'
        transFif = TRANS_DIR / ('sub-' + subject_id + '-trans.fif')
        epochFif = EPOCH_DIR + subject_id + fifFile

    return epochFif, transFif, bemFif


def run_csc(X, **cdl_params):
    """

    Parameters
    ----------
    X : numpy.ndarray
        the data to run the CSC on

    cdl_params : dict
        dictionary of CSC parameters, such as 'n_atoms', 'n_times_atoms', etc.

    Returns
    -------
    cdl_model

    z_hat_

    """
    print('Computing CSC')

    cdl_params = dict(cdl_params)
    n_splits = cdl_params.pop('n_splits', 1)
    use_batch_cdl = cdl_params.pop('use_batch_cdl', False)
    if use_batch_cdl:
        cdl_model = BatchCDL(**cdl_params)
    else:
        cdl_model = GreedyCDL(**cdl_params)

    if n_splits > 1:
        X_splits = split_signal(X, n_splits=n_splits, apply_window=True)
        X = X[None, :]
    else:
        X_splits = X.copy()

    # Fit the model and learn rank1 atoms
    print('Running CSC')
    cdl_model.fit(X_splits)

    z_hat_ = cdl_model.transform(X)
    return cdl_model, z_hat_


def get_subject_info(participants_file, subject_id, verbose=False):
    """For a given subject id, return its age and sex found in the csv
    containing all participant info.

    Parameters
    ----------
    participants_file : str | Pathlib instance
        Path to csv containing all participants info

    Returns
    -------
    age : float
        the age of the considered participant

    sex : str
        the sex (MALE or FEMALE) of the considered participant
    """

    # get age and sex of the subject
    participants = pd.read_csv(participants_file, sep='\t', header=0)
    age, sex = participants[participants['participant_id']
                            == 'sub-' + str(subject_id)][['age', 'sex']].iloc[0]
    if verbose:
        print(f'Subject ID: {subject_id}, {str(age)} year old {sex}')

    return age, sex


def get_atom_df(results_dir, participants_file):
    """ Create a pandas.DataFrame where each row is an atom, and columns are
    crutial informations, such a the subject id, its u and v vectors as well
    as the participant age and sex.

    Parameters
    ----------
    results_dir : Pathlib instance
        Path to all participants CSC pickled results

    participants_file : str | Pathlib instance
        Path to csv containing all participants info

    Returns
    -------
    pandas.DataFrame
    """

    subject_dirs = [f for f in results_dir.iterdir() if not f.is_file()]

    df = pd.DataFrame()
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        # get participant info
        age, sex = get_subject_info(participants_file, subject_id)
        base_row = {'subject_id': subject_id, 'age': age, 'sex': sex}
        # get participant CSC results
        file_name = results_dir / subject_id / 'CSCraw_0.5s_20atoms.pkl'
        if not file_name.exists():
            print(f"No such file or directory: {file_name}")
            break

        # load CSC results
        cdl_model, _, allZ, _ = pickle.load(open(file_name, "rb"))

        # make epoch and compute dipole fit
        epochFif, transFif, bemFif = get_paths(subject_id, dal=True)
        # Read in epochs for task data
        epochs = mne.read_epochs(epochFif)
        epochs.pick_types(meg='grad')
        info = epochs.info
        cov = mne.compute_covariance(epochs)

        # Make an evoked object with all atom topographies for dipole fitting
        evoked = mne.EvokedArray(cdl_model.u_hat_.T, info)
        # compute dipole fit
        dip = mne.fit_dipole(evoked, cov, bemFif,
                             transFif,  verbose=False)[0]

        for kk, (u, v) in enumerate(zip(cdl_model.u_hat_, cdl_model.v_hat_)):
            gof = dip.gof[kk]
            pos = dip.pos[kk]  # 3-element list (index to x, y, z)
            ori = dip.ori[kk]  # 3-element list (index to x, y, z)
            # calculate the percent change in activation between different
            # phases of movement
            pre = allZ[:, kk, 68:218]  # -1.25 to -0.25 sec (150 samples)
            move = allZ[:, kk, 218:293]  # -0.25 to 0.25 sec (75 samples)
            # 0.25 to 1.25 sec (150 samples)
            post = allZ[:, kk, 293:443]

            pre_sum = np.sum(pre)
            move_sum = np.sum(move)
            post_sum = np.sum(post)
            # multiply by 2 for movement phase because there are half as many samples
            z1 = (pre_sum-move_sum*2)/pre_sum
            z2 = (post_sum-move_sum*2)/post_sum
            z3 = (post_sum-pre_sum)/post_sum
            # update dataframe
            new_row = {**base_row, 'atom_id': int(kk), 'u_hat': u, 'v_hat': v,
                       'Dipole GOF': gof, 'Dipole Pos x': pos[0],
                       'Dipole Pos y': pos[1], 'Dipole Pos z': pos[2],
                       'Dipole Ori x': ori[0], 'Dipole Ori y': ori[1],
                       'Dipole Ori z': ori[2],
                       'Pre-Move Change': z1,
                       'Post-Move Change': z2,
                       'Post-Pre Change': z3}
            df = df.append(new_row, ignore_index=True)

    return df


OUTPUT_DIR = '/media/NAS/lpower/CSC/results/u_'


def correlation_clustering_atoms(atom_df, threshold=0.4, output_dir=OUTPUT_DIR):
    """

    Parameters
    ----------
    threshold : float
        threshold to create new groups

    Returns
    -------
    groupSummary, atomGroups (and save them XXX)

    """

    # XXX exclude 'bad' subjects (single slustering operation)

    # XXX make it read a pre-saved file
    exclude_subs = ['CC420061', 'CC121397', 'CC420396', 'CC420348', 'CC320850',
                    'CC410325', 'CC121428', 'CC110182', 'CC420167', 'CC420261',
                    'CC322186', 'CC220610', 'CC221209', 'CC220506', 'CC110037',
                    'CC510043', 'CC621642', 'CC521040', 'CC610052', 'CC520517',
                    'CC610469', 'CC720497', 'CC610292', 'CC620129', 'CC620490']

    atom_df = atom_df[~atom_df['subject_id'].isin(exclude_subs)]
    # Calculate the correlation coefficient between all atoms
    u_vector_list = np.asarray(atom_df['u_hat'].values)
    v_vector_list = np.asarray(atom_df['v_hat'].values)

    v_coefs = []
    for v in v_vector_list:
        for v2 in v_vector_list:
            coef = np.max(ss.correlate(v, v2))
            v_coefs.append(coef)
    v_coefs = np.asarray(v_coefs)
    v_coefs = np.reshape(v_coefs, (10760, 10760))

    u_coefs = np.corrcoef(u_vector_list, u_vector_list)[0:10760][0:10760]

    threshold_summary = pd.DataFrame(
        columns=['Threshold', 'Number of Groups', 'Number of Top Groups'])

    # Set parameters
    u_thresh = threshold
    v_thresh = threshold

    atomNum = 0
    groupNum = 0
    unique = True

    # Make atom groups array to keep track of the group that each atom belongs to
    atomGroups = pd.DataFrame(
        columns=['subject_id', 'atom_id', 'Index', 'Group number'])

    for ii, row in atom_df.iterrows():
        print(row)
        subject_id, atom_id = row.subject_id, row.atom_id

        max_corr = 0
        max_group = 0

        # Loops through the existing groups and calculates the atom's average correlation to that group
        for group in range(0, groupNum + 1):
            gr_atoms = atomGroups[atomGroups['Group number'] == group]
            inds = gr_atoms['Index'].tolist()
            u_groups = []
            v_groups = []

            # Find the u vector and correlation coefficient comparing the current atom to each atom in the group
            for ind2 in inds:
                u_coef = u_coefs[ii][ind2]
                u_groups.append(u_coef)

                v_coef = v_coefs[ii][ind2]
                v_groups.append(v_coef)

            # average across u and psd correlation coefficients in that group
            u_groups = abs(np.asarray(u_groups))
            avg_u = np.mean(u_groups)

            v_groups = abs(np.asarray(v_groups))
            avg_v = np.mean(v_groups)

            # check if this group passes the thresholds
            if (avg_u > u_thresh) & (avg_v > v_thresh):
                unique = False
                # If it does, also check if this is the highest cumulative correlation so far
                if (avg_u + avg_v) > max_corr:
                    max_corr = (avg_u + avg_v)
                    max_group = group

        # If the atom was similar to at least one group, sorts it into the group that it had the highest cumulative correlation to
        if (unique == False):
            groupDict = {'subject_id': subject_id, 'atom_id': atom_id,
                         'Index': ii, 'Group number': max_group}

        # If a similar group is not found, a new group is create and the current atom is added to that group
        elif (unique == True):
            groupNum += 1
            print(groupNum)
            groupDict = {'subject_id': subject_id, 'atom_id': atom_id,
                         'Index': ii, 'Group number': groupNum}

        # Add to group dataframe and reset unique boolean
        atomGroups = atomGroups.append(groupDict, ignore_index=True)
        unique = True

        # Summary statistics for the current dataframe:

    # Number of distinct groups
    groups = atomGroups['Group number'].tolist()
    groups = np.asarray(groups)
    numGroups = len(np.unique(groups))

    # Number of atoms and subjects per group
    numAtoms_list = []
    numSubs_list = []

    for un in np.unique(groups):
        numAtoms = len(np.where(groups == un)[0])
        numAtoms_list.append(numAtoms)

        groupRows = atomGroups[atomGroups['Group number'] == un]
        sub_list = np.asarray(groupRows['Subject ID'].tolist())
        numSubs = len(np.unique(sub_list))
        numSubs_list.append(numSubs)

    numAtoms_list = np.asarray(numAtoms_list)
    meanAtoms = np.mean(numAtoms_list)
    stdAtoms = np.std(numAtoms_list)

    print("Number of groups:")
    print(numGroups)

    print("Average number of atoms per group:")
    print(str(meanAtoms) + " +/- " + str(stdAtoms))

    groupSummary = pd.DataFrame(
        columns=['Group Number', 'Number of Atoms', 'Number of Subjects'])
    groupSummary['Group Number'] = np.unique(groups)
    groupSummary['Number of Atoms'] = numAtoms_list
    groupSummary['Number of Subjects'] = numSubs_list

    numSubs_list = np.asarray(numSubs_list)
    topGroups = len(np.where(numSubs_list >= 12)[0])
    threshold_dict = {'Threshold': threshold,
                      'Number of Groups': numGroups, 'Number of Top Groups': topGroups}
    threshold_summary = threshold_summary.append(
        threshold_dict, ignore_index=True)

    # Save group summary dataframe
    csv_dir = output_dir + \
        str(u_thresh) + '_v_' + str(v_thresh) + '_groupSummary.csv'
    groupSummary.to_csv(csv_dir)

    # Save atomGroups to dataframe
    csv_dir = output_dir + \
        str(u_thresh) + '_v_' + str(v_thresh) + '_atomGroups.csv'
    atomGroups.to_csv(csv_dir)

    return groupSummary, atomGroups


def culstering_cah_kmeans(df, data_columns='all', n_clusters=6):
    """Compute a CAH and k-means clustering

    """
    if data_columns == 'all':
        data = np.array(df)
    else:
        data = np.array(df[data_columns])
    # CAH clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                         affinity='euclidean',
                                         linkage='ward')
    clustering.fit(data)
    df['labels_cah'] = clustering.labels_
    # k-means clustering
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    df['labels_kmeans'] = kmeans.labels_

    return df


def reconstruct_class_signal(df, results_dir):
    """ Reonstruct the signal for all atoms in the given dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe where each row is an atom, and has at least
        the folowing columns :
            subject_id : the participant id associated with the atom
            atom_id : the atom id

    results_dir : Pathlib instance
        Path to all participants CSC pickled results

    Returns
    -------
    X : array-like
        the reconstructed signal

    n_times_atom : int
        the minimum number of timestamps per atom, accross all atoms in the
        input dataframe
    """

    Z_temp = []
    D_temp = []
    min_n_times_valid = np.inf
    for subject_id in set(df['subject_id'].values):
        file_name = results_dir / subject_id / 'CSCraw_0.5s_20atoms.pkl'
        cdl_model, info, _, _ = pickle.load(open(file_name, "rb"))
        atom_idx = df[df['subject_id'] ==
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

    n_times_atom = D.shape[-1]

    X = construct_X_multi(Z, D)

    return X, n_times_atom


def get_df_mean(df, col_label='Group number', cdl_params=CDL_PARAMS,
                results_dir=RESULT_DIR, n_jobs=6):
    """

    Parameters
    ----------
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
        the CDL parameters to use to compute the mean atom.
        By default, use GreedyCDL, to use BatchCDL, ensure that
        cdl_params['use_batch_cdl'] = True

    results_dir : Pathlib instance
        Path to all participants CSC pickled results

    n_jobs : int
        number of concurrently running jobs
        default is 6

    Returns
    -------
    df_mean : pandas.DataFrame
        columns:
            col_label : clustering label
            u_hat, v_hat : spatial and temporal pattern of the mean atom
            z_hat : activation vector of the mean atom
    """

    # ensure that only one recurring pattern will be extracted
    cdl_params['n_atoms'] = 1
    cdl_params['n_splits'] = 1

    def procedure(label):
        # Reconstruct signal for a given class
        X, n_times_atom = reconstruct_class_signal(
            df=df[df[col_label] == label], results_dir=results_dir)
        cdl_params['n_times_atom'] = n_times_atom
        cdl_model, z_hat = run_csc(X, **cdl_params)
        # append dataframe
        new_row = {col_label: label,
                   'u_hat': cdl_model.u_hat_[0],
                   'v_hat': cdl_model.v_hat_[0],
                   'z_hat': z_hat,
                   'n_times_atom': n_times_atom}

        return new_row

    new_rows = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(procedure)(label) for label in set(df[col_label].values))

    df_mean = pd.DataFrame()
    for new_row in new_rows:
        df_mean = df_mean.append(new_row, ignore_index=True)

    df_mean.rename(columns={col_label: 'label'}, inplace=True)
    df_mean.to_csv(RESULT_DIR / 'df_mean_atom.csv')

    return df_mean


if __name__ == '__main__':
    atomData = pd.read_csv('atomData.csv')

    atomData.rename(columns={'Subject ID': 'subject_id',
                             'Atom number': 'atom_id'},
                    inplace=True)

    participants = pd.read_csv("participants.tsv", sep='\t', header=0)
    participants['subject_id'] = participants['participant_id'].apply(
        lambda x: x[4:])

    columns = ['subject_id', 'atom_id', 'Dipole GOF',
               'Dipole Pos x', 'Dipole Pos y', 'Dipole Pos z',
               'Dipole Ori x', 'Dipole Ori y', 'Dipole Ori z', 'Focal',
               'Pre-Move Change', 'Post-Move Change',
               'Post-Pre Change', 'Movement-related', 'Rebound']

    atom_df_temp = pd.merge(atomData[columns], participants[[
        'subject_id', 'age', 'sex', 'hand']], on="subject_id")

    results_dir = Path('./results_csc')
    subject_dirs = [f for f in results_dir.iterdir() if not f.is_file()]

    df = pd.DataFrame()
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        base_row = {'subject_id': subject_id}
        # get participant CSC results
        file_name = results_dir / subject_id / 'CSCraw_0.5s_20atoms.pkl'
        if not file_name.exists():
            print(f"No such file or directory: {file_name}")
            break

        # load CSC results
        cdl_model, _, allZ, _ = pickle.load(open(file_name, "rb"))

        for kk, (u, v) in enumerate(zip(cdl_model.u_hat_, cdl_model.v_hat_)):
            new_row = {**base_row, 'atom_id': int(kk), 'u_hat': u, 'v_hat': v}
            df = df.append(new_row, ignore_index=True)

    atom_df = pd.merge(atom_df_temp, df, how="left",
                       on=["subject_id", "atom_id"])
    atom_df.to_csv('atom_df.csv')

    atom_df.rename(columns={col: col.lower().replace(' ', '_')
                            for col in atom_df.columns},
                   inplace=True)

    atom_df.to_csv('atom_df.csv')
