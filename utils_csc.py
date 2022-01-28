"""
Utils scripts for utils functions 
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from joblib import Memory, Parallel, delayed

import mne
from alphacsc import BatchCDL, GreedyCDL
from alphacsc.utils.signal import split_signal
from alphacsc.utils.convolution import construct_X_multi

# Paths for Cam-CAN dataset
DATA_DIR = Path("/storage/store/data/")
BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"  # Root path to raw BIDS files
PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"


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

    df = pd.DataFrame
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        # get participant info
        age, sex = get_subject_info(participants_file, subject_id)
        base_row = {'subject_id': subject_id,
                    'age': age,
                    'sex': sex}
        # get participant CSC results
        file_name = results_dir / subject_id / 'CSCraw_0.5s_20atoms.pkl'
        cdl_model = pickle.load(open(file_name, "rb"))[0]
        for i, (u, v) in enumerate(zip(cdl_model.u_hat_, cdl_model.v_hat_)):
            new_row = {**base_row, 'atom_id': i, 'u_hat': u, 'v_hat': v}
            df = df.append(new_row, ignore_index=True)

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


def get_df_mean(df, col_label, cdl_params, results_dir, n_jobs=6):
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
                   'z_hat': z_hat}

        return new_row

    new_rows = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(procedure)(label) for label in set(df[col_label].values))

    df_mean = pd.DataFrame()
    for new_row in new_rows:
        df_mean = df_mean.append(new_row, ignore_index=True)

    return df_mean
