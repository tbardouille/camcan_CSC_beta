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
from mne_bids import BIDSPath, read_raw_bids

from alphacsc import BatchCDL, GreedyCDL
from alphacsc.utils.signal import split_signal
from alphacsc.utils.convolution import construct_X_multi

from config import CDL_PARAMS, SUBJECT_IDS, get_paths, get_cdl_pickle_name
from config import BIDS_ROOT, SSS_CAL_FILE, CT_SPARSE_FILE
from config import RESULTS_DIR, PARTICIPANTS_FILE, N_JOBS


def get_raw(subject_id, ch_type='grad', sfreq=150.):
    """


    """

    if subject_id[:4] == 'sub-':
        subject_id = subject_id.split('-')[1]

    # Read raw data from BIDS file
    bp = BIDSPath(
        root=BIDS_ROOT,
        subject=subject_id,
        task="smt",
        datatype="meg",
        extension=".fif",
        session="smt",
    )
    raw = read_raw_bids(bp)

    # Preprocess data
    raw.load_data()
    raw.filter(l_freq=None, h_freq=125)
    raw.notch_filter([50, 100])
    raw = mne.preprocessing.maxwell_filter(raw, calibration=SSS_CAL_FILE,
                                           cross_talk=CT_SPARSE_FILE,
                                           st_duration=10.0)
    # Now deal with Epochs
    all_events, all_event_id = mne.events_from_annotations(raw)
    # all_event_id = {'audiovis/1200Hz': 1, 'audiovis/300Hz': 2, 'audiovis/600Hz': 3,
    #                 'button': 4, 'catch/0': 5, 'catch/1': 6}

    metadata, events, event_id = mne.epochs.make_metadata(
        events=all_events, event_id=all_event_id,
        tmin=-3., tmax=0, sfreq=raw.info['sfreq'],
        row_events=['button'], keep_last=['audiovis'])

    epochs = mne.Epochs(
        raw, events, event_id, metadata=metadata,
        tmin=-1.7, tmax=1.7,
        baseline=(-1.25, -1.0),
        preload=True, verbose=False
    )

    # "good" button events: button event is at most one sec. after an audiovis
    # event, and with at least 3 sec. between 2 button events.
    epochs = epochs["event_name == 'button' and audiovis > -1. and button == 0."]

    # Band-pass filter the data to a range of interest
    raw.pick([ch_type, 'stim'])
    raw.filter(l_freq=2, h_freq=45)
    raw, events = raw.resample(
        sfreq, npad='auto', verbose=False, events=epochs.events)

    return raw, events


def run_csc(X, **cdl_params):
    """Run a CSC model on a given signal X.

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


def get_subject_info(subject_id, participants_file=PARTICIPANTS_FILE,
                     verbose=False):
    """For a given subject id, return its age, sex and hand found in the csv
    containing all participant info.

    Parameters
    ----------
    subject_id : str
        the subject id

    participants_file : str | Pathlib instance
        Path to csv containing all participants info

    verbose : bool
        if True, will print obtained info

    Returns
    -------
    age : float
        the age of the considered participant

    sex : str
        the sex (MALE | FEMALE) of the considered participant

    hand 
    """

    # get age and sex of the subject
    participants = pd.read_csv(participants_file, sep='\t', header=0)
    age, sex, hand = participants[participants['participant_id']
                                  == 'sub-' + str(subject_id)][['age', 'sex', 'hand']].iloc[0]
    if verbose:
        print(f'Subject ID: {subject_id}, {str(age)} year old {sex}')

    return age, sex, hand


def get_subject_dipole(subject_id, cdl_model=None, info=None):
    """Compute the atoms' dipoles for a subject for a pre-computed CDL model.

    Parameters
    ----------
    subject_id : str
        the subject id

    cdl_model : alphacsc.ConvolutionalDictionaryLearning instance

    info : mne.Info instance


    Returns
    -------
    dip : mne.Dipole instance

    """
    epochFif, transFif, bemFif = get_paths(subject_id)
    if (cdl_model is None) or (info is None):
        # get participant CSC results
        file_name = RESULTS_DIR / subject_id / get_cdl_pickle_name()
        if not file_name.exists():
            print(f"No such file or directory: {file_name}")
            return
        # load CSC results
        cdl_model, info, _, _ = pickle.load(open(file_name, "rb"))
    # compute noise covariance
    cov = mne.make_ad_hoc_cov(info)
    # select only grad channels
    meg_indices = mne.pick_types(info, meg='grad')
    info = mne.pick_info(info, meg_indices)
    evoked = mne.EvokedArray(cdl_model.u_hat_.T, info)
    # compute dipole fit
    dip = mne.fit_dipole(evoked, cov, str(bemFif), str(transFif), n_jobs=6,
                         verbose=False)[0]

    # in DAL code
    # # Fit a dipole for each atom
    # # Read in epochs for task data
    # epochs = mne.read_epochs(epochFif)
    # epochs.pick_types(meg='grad')
    # cov = mne.compute_covariance(epochs)
    # info = epochs.info

    # # Make an evoked object with all atom topographies for dipole fitting
    # evoked = mne.EvokedArray(cdl_model.u_hat_.T, info)

    # # Fit dipoles
    # dip = mne.fit_dipole(evoked, cov, bemFif, transFif, verbose=False)[0]

    return dip


def get_atoms_info(subject_id, results_dir=RESULTS_DIR):
    """For a given subject, return a list of dictionary containing all atoms'
    informations (subject info, u and v vectors, dipole informations, changes
    in activation before and after button press).

    Parameters
    ----------
    subject_id : str
        the subject id

    results_dir : Pathlib instance
        Path to all participants CSC pickled results

    Returns
    -------
    new_rows : list of dict
    """

    # get participant CSC results
    file_name = results_dir / subject_id / get_cdl_pickle_name()
    if not file_name.exists():
        print(f"No such file or directory: {file_name}")
        return

    # load CSC results
    cdl_model, info, allZ, _ = pickle.load(open(file_name, "rb"))

    # get informations about the subject
    age, sex, hand = get_subject_info(subject_id, PARTICIPANTS_FILE)
    base_row = {'subject_id': subject_id, 'age': age, 'sex': sex, 'hand': hand}
    # get informations about atoms
    dip = get_subject_dipole(subject_id, cdl_model, info=info)

    new_rows = []
    for kk, (u, v) in enumerate(zip(cdl_model.u_hat_, cdl_model.v_hat_)):
        gof, pos, ori = dip.gof[kk], dip.pos[kk], dip.ori[kk]

        # calculate the percent change in activation between different phases of movement
        # -1.25 to -0.25 sec (150 samples)
        pre_sum = np.sum(allZ[:, kk, 68:218])
        # -0.25 to 0.25 sec (75 samples)
        move_sum = np.sum(allZ[:, kk, 218:293])
        # 0.25 to 1.25 sec (150 samples)
        post_sum = np.sum(allZ[:, kk, 293:443])

        # multiply by 2 for movement phase because there are half as many samples
        z1 = (pre_sum - 2 * move_sum) / pre_sum
        z2 = (post_sum - 2 * move_sum) / post_sum
        z3 = (post_sum - pre_sum) / post_sum

        new_rows.append({
            **base_row, 'atom_id': kk, 'u_hat': u, 'v_hat': v, 'dipole_gof': gof,
            'dipole_pos_x': pos[0], 'dipole_pos_y': pos[1], 'dipole_pos_z': pos[2],
            'dipole_ori_x': ori[0], 'dipole_ori_y': ori[1], 'dipole_ori_z': ori[2],
            'pre-move_change': z1, 'post-move_change': z2, 'post-pre_change': z3,
            'focal': (gof >= 95), 'rebound': (z3 >= 0.1),
            'movement_related': (z1 >= 0. and z2 >= 0.6)
        })

    return new_rows


def get_atom_df(subject_ids=SUBJECT_IDS, results_dir=RESULTS_DIR, save=True):
    """ Create a pandas.DataFrame where each row is an atom, and columns are
    crutial informations, such a the subject id, its u and v vectors as well
    as the participant age and sex.

    Parameters
    ----------
    subject_ids : list of str
        list of subject ids to which we want to collect their atoms' info

    results_dir : Pathlib instance
        Path to all participants CSC pickled results

    save : bool
        if True, save output dataframe as csv
        defaults to True

    Returns
    -------
    pandas.DataFrame
    """

    new_rows = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(get_atoms_info)(this_subject_id) for this_subject_id in subject_ids)

    df = pd.DataFrame()
    for this_new_row in new_rows:
        df = df.append(this_new_row, ignore_index=True)

    if save:
        df.to_csv(results_dir / 'all_atoms_info.csv')
        pickle.dump(df, open(results_dir / 'all_atoms_info.pkl', "wb"))

    return df


def correlation_clustering_atoms(atom_df, threshold=0.4,
                                 output_dir=RESULTS_DIR):
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

        # Loops through the existing groups and calculates the atom's average
        # correlation to that group
        for group in range(0, groupNum + 1):
            gr_atoms = atomGroups[atomGroups['Group number'] == group]
            inds = gr_atoms['Index'].tolist()
            u_groups = []
            v_groups = []

            # Find the u vector and correlation coefficient comparing the
            # current atom to each atom in the group
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
                # If it does, also check if this is the highest cumulative
                # correlation so far
                if (avg_u + avg_v) > max_corr:
                    max_corr = (avg_u + avg_v)
                    max_group = group

        # If the atom was similar to at least one group, sorts it into the
        # group that it had the highest cumulative correlation to
        if (unique == False):
            groupDict = {'subject_id': subject_id, 'atom_id': atom_id,
                         'Index': ii, 'Group number': max_group}

        # If a similar group is not found, a new group is create and the
        # current atom is added to that group
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
                      'Number of Groups': numGroups,
                      'Number of Top Groups': topGroups}
    threshold_summary = threshold_summary.append(
        threshold_dict, ignore_index=True)

    # Save group summary dataframe
    csv_dir = output_dir + \
        'u_' + str(u_thresh) + '_v_' + str(v_thresh) + '_groupSummary.csv'
    groupSummary.to_csv(csv_dir)

    # Save atomGroups to dataframe
    csv_dir = output_dir + \
        'u_' + str(u_thresh) + '_v_' + str(v_thresh) + '_atomGroups.csv'
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
        file_name = results_dir / subject_id / get_cdl_pickle_name()
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
                results_dir=RESULTS_DIR, n_jobs=N_JOBS):
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

    Returns
    -------
    df_mean : pandas.DataFrame
        columns:
            col_label : clustering label
            u_hat, v_hat : spatial and temporal pattern of the mean atom
            z_hat : activation vector of the mean atom
    """

    # ensure that only one recurring pattern will be extracted
    cdl_params.update(n_atoms=1, n_splits=1)

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

    print(df_mean)

    df_mean.rename(columns={col_label: 'label'}, inplace=True)
    df_mean.to_csv(results_dir / 'df_mean_atom.csv')

    return df_mean


def complete_existing_df(atomData, results_dir=RESULTS_DIR):
    """

    """
    atomData = pd.read_csv('atomData.csv')

    atomData.rename(columns={'Subject ID': 'subject_id',
                             'Atom number': 'atom_id'},
                    inplace=True)

    # participants = pd.read_csv("participants.tsv", sep='\t', header=0)
    participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t', header=0)
    participants['subject_id'] = participants['participant_id'].apply(
        lambda x: x[4:])

    columns = ['subject_id', 'atom_id', 'Dipole GOF',
               'Dipole Pos x', 'Dipole Pos y', 'Dipole Pos z',
               'Dipole Ori x', 'Dipole Ori y', 'Dipole Ori z', 'Focal',
               'Pre-Move Change', 'Post-Move Change',
               'Post-Pre Change', 'Movement-related', 'Rebound']

    atom_df_temp = pd.merge(atomData[columns], participants[[
        'subject_id', 'age', 'sex', 'hand']], how="left", on="subject_id")

    subject_dirs = [f for f in results_dir.iterdir() if not f.is_file()]

    df = pd.DataFrame()
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        base_row = {'subject_id': subject_id}
        # get participant CSC results
        file_name = results_dir / subject_id / get_cdl_pickle_name()
        if not file_name.exists():
            print(f"No such file or directory: {file_name}")
            break

        # load CSC results
        cdl_model, _, _, _ = pickle.load(open(file_name, "rb"))

        for kk, (u, v) in enumerate(zip(cdl_model.u_hat_, cdl_model.v_hat_)):
            new_row = {**base_row, 'atom_id': int(kk), 'u_hat': u, 'v_hat': v}
            df = df.append(new_row, ignore_index=True)

    atom_df = pd.merge(atom_df_temp, df, how="left",
                       on=["subject_id", "atom_id"])
    atom_df.rename(columns={col: col.lower().replace(' ', '_')
                            for col in atom_df.columns},
                   inplace=True)
    atom_df.to_csv('all_atoms_info.csv')
    pickle.dump(atom_df, open('all_atoms_info.pkl', "wb"))

    return atom_df


if __name__ == '__main__':
    atomData = pd.read_csv(RESULTS_DIR / 'atomData.csv')
    all_atoms_info = complete_existing_df(atomData)
