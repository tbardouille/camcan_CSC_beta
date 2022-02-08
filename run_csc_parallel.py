"""For all subjects in Cam-CAN folder, run CSC and compute informations. Save
the final result in a dataframe and pickle.
"""

import numpy as np
import pandas as pd
import pickle
import json
from joblib import Memory, Parallel, delayed

from alphacsc.viz.epoch import make_epochs

from config import (RESULTS_DIR, SUBJECT_IDS, N_JOBS,
                    CDL_PARAMS, EXP_PARAMS, get_cdl_pickle_name)
from utils_csc import get_raw, run_csc, get_atom_df

mem = Memory('.')


def procedure(subject_id):
    """

    """
    # get preprocessed raw and events
    raw, events = get_raw(subject_id)
    # run multivariate CSC
    cdl_model, z_hat_ = mem.cache(run_csc)(
        X=raw.get_data(picks=['meg']), **CDL_PARAMS)
    # events here are only "good" button events
    events_no_first_samp = events.copy()
    events_no_first_samp[:, 0] -= raw.first_samp
    info = raw.info.copy()
    info.update(events=events_no_first_samp, event_id=None)

    allZ = make_epochs(
        z_hat_, info, t_lim=(-1.7, 1.7), n_times_atom=CDL_PARAMS['n_times_atom'])

    # save CDL results in pickle, as well as experiment parameters
    subject_res_dir = RESULTS_DIR / subject_id
    subject_res_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump([cdl_model, raw.info, allZ, z_hat_],
                open(subject_res_dir / get_cdl_pickle_name(), "wb"))
    with open(subject_res_dir / 'exp_params', 'w') as fp:
        json.dump([EXP_PARAMS, CDL_PARAMS], fp, sort_keys=True, indent=4)

    return None


Parallel(n_jobs=N_JOBS, verbose=1)(
    delayed(procedure)(this_subject_id) for this_subject_id in SUBJECT_IDS)

# from the CDL results, save in a dataframe info about all atoms
atom_df = get_atom_df(SUBJECT_IDS, results_dir=RESULTS_DIR, save=True)
