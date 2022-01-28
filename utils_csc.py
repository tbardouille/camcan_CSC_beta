"""
Utils scripts for utils functions 
"""
import pandas as pd
from pathlib import Path
import pickle

from alphacsc import BatchCDL, GreedyCDL
from alphacsc.utils.signal import split_signal

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
    as the participant age

    Parameters
    ----------
    results_dir : Pathlib instance
        Path to all participants CSC pickled results

    participants_file : Pathlib instance
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
        cdl_model, info, _, _ = pickle.load(open(file_name, "rb"))
        # u_hat_, v_hat_ = cdl_model.u_hat_, cdl_model.v_hat_
        for i, (u, v) in enumerate(zip(cdl_model.u_hat_, cdl_model.v_hat_)):
            new_row = {**base_row, 'atom_id': i, 'u_hat': u, 'v_hat': v}
            df = df.append(new_row, ignore_index=True)

    return df


def reconstruct_signal_from_atoms(cdl_model, z_hat_):
    """

    """
